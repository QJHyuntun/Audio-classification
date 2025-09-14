"""
train_Feature.py

通用训练脚本：支持三种特征提取网络互换（FAEN, MBVAE, CWAFE）
假设你已将三种模型定义放入 models.py 并导出了类名：
    - MBVAE
    - CWAFE
    - FAEN

依赖:
    torch, torchaudio, transformers (可选, 用于CWAFE), sklearn

使用示例:
    python train_Feature.py --model faen --input_type mel --data_csv data/train.csv --epochs 30 --batch_size 16
    python train_Feature.py --model cwafe --input_type wave --data_csv data/train.csv --epochs 20
    python train_Feature.py --model mbvae --input_type mbands --data_csv data/train_bands.csv

data_csv 格式 (示例):
    filepath,label
    ./wav/001.wav,0
    ./wav/002.wav,2
"""

import os
import argparse
import time
import random
from pathlib import Path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from sklearn.metrics import accuracy_score

# ---------------------------
# 尝试导入你之前定义的模型（若放在 models.py )
# ---------------------------
try:
    from model.MB_VAE import MB_VAE
    from model.CWAFE import CWAFE
    from model.FAEN import FAEN
except Exception as e:
    print("Warning: failed to import models from models.py — 请将 MBVAE, CWAFE, FAEN 类放到 models.py 中。")
    print("Import error:", e)
    raise


# ---------------------------
# 数据集：支持三种 input_type
#    - 'wave' : returns waveform tensor (B, L) float32
#    - 'mel'  : returns mel-spectrogram (1, n_mels, time_frames)
#    - 'mbands': returns list of 3-band spectrogram tensors for MB-VAE,
#                each shaped (1, freq_bins_i, time_frames)
# 说明：脚本使用 torchaudio.load + torchaudio.transforms.MelSpectrogram
# ---------------------------
class AudioDataset(Dataset):
    def __init__(self, csv_path, input_type="mel", sr=16000, n_mels=128,
                 mband_splits=(0.15, 0.60, 0.25), max_duration=None):
        """
        csv_path: csv with columns 'filepath','label'
        input_type: 'wave'|'mel'|'mbands'
        mband_splits: used only when input_type == 'mbands' - proportions of freq bins
        """
        import csv
        self.rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append((r['filepath'], int(r['label'])))
        self.input_type = input_type
        self.sr = sr
        self.n_mels = n_mels
        self.mband_splits = mband_splits
        self.max_duration = max_duration  # seconds, optional truncation/pad

        # mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=512, hop_length=256, n_mels=n_mels
        )

    def _load_wave(self, path):
        wav, sr = torchaudio.load(path)  # shape (channels, L)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        else:
            wav = wav
        wav = wav.squeeze(0)  # (L,)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        # optional trimming/padding to max_duration
        if self.max_duration is not None:
            max_len = int(self.max_duration * self.sr)
            if wav.shape[0] > max_len:
                wav = wav[:max_len]
            elif wav.shape[0] < max_len:
                pad = max_len - wav.shape[0]
                wav = F.pad(wav, (0, pad))
        return wav

    def wave_to_mel(self, wav):
        # wav shape (L,)
        mel = self.mel_transform(wav.unsqueeze(0))  # (1, n_mels, T)
        mel = torchaudio.functional.amplitude_to_DB(mel, multiplier=20, amin=1e-5, db_multiplier=0)  # dB
        # normalize to -1..1 roughly by dividing dynamic range (simple)
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        return mel  # (1, n_mels, T)

    def split_mbands(self, mel):
        """
        mel: (1, n_mels, T)
        split along frequency axis into 3 tensors (1, f1, T), (1, f2, T), (1, f3, T)
        by proportions mband_splits.
        """
        _, n_mels, T = mel.shape
        a, b, c = self.mband_splits
        k1 = max(1, int(a * n_mels))
        k2 = max(1, int((a + b) * n_mels))
        # ensure covers
        k2 = min(k2, n_mels - 1)
        band1 = mel[:, :k1, :]
        band2 = mel[:, k1:k2, :]
        band3 = mel[:, k2:, :]
        return [band1, band2, band3]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        wave = self._load_wave(path)
        if self.input_type == "wave":
            return wave, label
        elif self.input_type == "mel":
            mel = self.wave_to_mel(wave)  # (1,n_mels,T)
            return mel, label
        elif self.input_type == "mbands":
            mel = self.wave_to_mel(wave)
            bands = self.split_mbands(mel)
            # for MBVAE in our earlier code, each band expected shape (C, F, T) where C=1
            return bands, label
        else:
            raise ValueError("unknown input_type: " + str(self.input_type))


# ---------------------------
# Trainer utilities
# ---------------------------
def collate_fn_factory(input_type):
    def collate_fn(batch):
        # batch is list of (input, label)
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        if input_type == "wave":
            waves = [b[0] for b in batch]
            lengths = [w.shape[0] for w in waves]
            max_len = max(lengths)
            stacked = torch.stack([F.pad(w, (0, max_len - w.shape[0])) for w in waves], dim=0)  # (B, L)
            return stacked, labels
        elif input_type == "mel":
            mels = [b[0] for b in batch]
            # pad time dimension
            max_t = max(m.shape[2] for m in mels)
            stacked = torch.stack([F.pad(m, (0, max_t - m.shape[2])) for m in mels], dim=0)  # (B,1,n_mels,T)
            return stacked, labels
        elif input_type == "mbands":
            # each sample: list of 3 bands tensors
            bands_list = [b[0] for b in batch]  # list length B, each is [b1,b2,b3]
            # pad each band's time axis independently
            B = len(bands_list)
            out_bands = []
            for band_idx in range(3):
                band_shapes = [bands_list[i][band_idx] for i in range(B)]
                max_t = max(x.shape[2] for x in band_shapes)
                padded = torch.stack([F.pad(x, (0, max_t - x.shape[2])) for x in band_shapes], dim=0)
                out_bands.append(padded)
            return out_bands, labels
    return collate_fn


# ---------------------------
# Training / Validation loops
# ---------------------------
def train_one_epoch(model, loader, optimizer, device, cfg, scaler=None):
    model.train()
    total_loss = 0.0
    preds_all = []
    labels_all = []
    for batch in loader:
        inputs, labels = batch
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward depending on model type
        if cfg.model == 'cwafe':
            wave = inputs.to(device)  # (B, L)
            logits = model(wave)
            loss = F.cross_entropy(logits, labels)
        elif cfg.model == 'faen':
            mel = inputs.to(device)  # (B,1,n_mels,T)
            logits = model(mel)
            loss = F.cross_entropy(logits, labels)
        elif cfg.model == 'mbvae':
            # inputs is list of 3 band tensors
            bands = [b.to(device) for b in inputs]  # each (B,1,F_i,T)
            logits, zs, mus, logvars = model(bands)
            # compute dual loss
            cls_loss = F.cross_entropy(logits, labels)
            # KL as in earlier MBVAE code
            kl_loss = 0.0
            for mu, logvar in zip(mus, logvars):
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                kl_loss += kl.mean()
            kl_loss = kl_loss / len(mus)
            loss = cls_loss + cfg.alpha_kl * kl_loss
        else:
            raise ValueError("Unknown model type")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        # predictions
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        preds_all.append(preds)
        labels_all.append(labels.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    acc = accuracy_score(labels_all, preds_all)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc


def validate(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            labels = labels.to(device)
            if cfg.model == 'cwafe':
                wave = inputs.to(device)
                logits = model(wave)
                loss = F.cross_entropy(logits, labels)
            elif cfg.model == 'faen':
                mel = inputs.to(device)
                logits = model(mel)
                loss = F.cross_entropy(logits, labels)
            elif cfg.model == 'mbvae':
                bands = [b.to(device) for b in inputs]
                logits, zs, mus, logvars = model(bands)
                cls_loss = F.cross_entropy(logits, labels)
                kl_loss = 0.0
                for mu, logvar in zip(mus, logvars):
                    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                    kl_loss += kl.mean()
                kl_loss = kl_loss / len(mus)
                loss = cls_loss + cfg.alpha_kl * kl_loss
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            preds_all.append(preds)
            labels_all.append(labels.detach().cpu().numpy())
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    acc = accuracy_score(labels_all, preds_all)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc


# ---------------------------
# Utility: save checkpoint & simple logging
# ---------------------------
def save_checkpoint(state, filename):
    torch.save(state, filename)


# ---------------------------
# Main run
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='faen', choices=['faen', 'mbvae', 'cwafe'])
    p.add_argument('--input_type', type=str, default='mel', choices=['wave', 'mel', 'mbands'])
    p.add_argument('--data_csv', type=str, required=True)
    p.add_argument('--val_csv', type=str, default=None)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--alpha_kl', type=float, default=0.1, help='KL weight for MBVAE')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Data
    train_ds = AudioDataset(cfg.data_csv, input_type=cfg.input_type)
    val_ds = AudioDataset(cfg.val_csv if cfg.val_csv else cfg.data_csv, input_type=cfg.input_type)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn_factory(cfg.input_type))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate_fn_factory(cfg.input_type))

    # Model selection
    device = torch.device(cfg.device)
    if cfg.model == 'faen':
        model = FAEN(num_classes=5, pretrained_backbone=False)  # adjust num_classes as needed
    elif cfg.model == 'cwafe':
        # CWAFE expects waveform input (B, L)
        model = CWAFE(num_classes=5, num_conformer_blocks=2, d_model=768, num_heads=8)
    elif cfg.model == 'mbvae':
        # MBVAE requires band_shapes argument; derive from dataset mel size by a dummy pass
        # We'll create dummy shapes based on dataset mel dims
        example_mel, _ = train_ds[0] if cfg.input_type != 'mbands' else (train_ds[0][0], train_ds[0][1])
        if cfg.input_type == 'mbands':
            # example_mel is list of 3 bands
            band_shapes = []
            for b in example_mel:
                _, f, t = b.shape
                band_shapes.append((1, f, t))
        else:
            # compute mel and split
            if cfg.input_type == 'wave':
                # create mel from first wave
                wave, _ = train_ds._load_wave(train_ds.rows[0][0]), 0
                mel = train_ds.wave_to_mel(wave)
            else:
                mel, _ = train_ds[0]
            bands = train_ds.split_mbands(mel)
            band_shapes = [(1, b.shape[1], b.shape[2]) for b in bands]
        model = MBVAE(band_shapes=band_shapes, latent_dim=32, num_classes=5, fusion="concat")
    else:
        raise ValueError("Unknown model selected")

    model = model.to(device)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, cfg)
        val_loss, val_acc = validate(model, val_loader, device, cfg)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f} | {elapsed:.1f}s")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
                'cfg': vars(cfg)
            }
            fname = os.path.join(cfg.save_dir, f"best_{cfg.model}.pt")
            save_checkpoint(ckpt, fname)
            print("Saved checkpoint:", fname)

    # final save
    final_path = os.path.join(cfg.save_dir, f"final_{cfg.model}.pt")
    save_checkpoint({'model_state': model.state_dict(), 'cfg': vars(cfg)}, final_path)
    print("Training finished. Final model saved to", final_path)


if __name__ == '__main__':
    main()
