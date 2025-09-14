import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse

# -----------------------------
# 参数配置
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--wav_dir', type=str, default='./wav_data', help='语音数据根目录')
parser.add_argument('--output_dir', type=str, default='./features', help='特征保存根目录')
parser.add_argument('--sr', type=int, default=16000, help='重采样频率')
parser.add_argument('--n_mels', type=int, default=128, help='Mel谱图通道数')
parser.add_argument('--n_fft', type=int, default=1024, help='STFT窗长度')
parser.add_argument('--hop_length', type=int, default=512, help='帧移长度')
args = parser.parse_args()

# -----------------------------
# 创建保存目录
# -----------------------------
spec_dir = Path(args.output_dir) / 'spectrogram'
acoustic_dir = Path(args.output_dir) / 'acoustic_seq'
mel_dir = Path(args.output_dir) / 'mel_spectrogram'

spec_dir.mkdir(parents=True, exist_ok=True)
acoustic_dir.mkdir(parents=True, exist_ok=True)
mel_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 遍历 wav 文件
# -----------------------------
for root, _, files in os.walk(args.wav_dir):
    for file in files:
        if file.endswith('.wav'):
            wav_path = os.path.join(root, file)
            y, sr = librosa.load(wav_path, sr=args.sr)
            filename = Path(file).stem

            # -----------------------------
            # 1. 频谱图特征（STFT）
            # -----------------------------
            stft = librosa.stft(y, n_fft=args.n_fft, hop_length=args.hop_length)
            stft_db = librosa.amplitude_to_db(np.abs(stft))
            np.save(spec_dir / f'{filename}_stft.npy', stft_db)

            # -----------------------------
            # 2. 声学序列特征（MFCC + Chroma）
            # -----------------------------
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=args.hop_length)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, hop_length=args.hop_length)
            acoustic_seq = np.vstack([mfcc, chroma])
            np.save(acoustic_dir / f'{filename}_acoustic.npy', acoustic_seq)

            # -----------------------------
            # 3. 梅尔谱图特征（Mel-Spectrogram）
            # -----------------------------
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=args.n_mels,
                                                      n_fft=args.n_fft, hop_length=args.hop_length)
            mel_db = librosa.power_to_db(mel_spec)
            np.save(mel_dir / f'{filename}_mel.npy', mel_db)

            print(f'Processed {file}')
