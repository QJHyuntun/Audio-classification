# train_msat.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SpeechDataset  # dataset.py
from MSAT import MSAT
from model.MB_VAE import MBVAE
from model.CWAFE import CWAFE
from model.FAEN import FAEN
from sklearn.metrics import accuracy_score

# -------------------------------
# 配置
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
lr = 1e-4
num_epochs = 50
num_classes = 5
embed_dim = 128
num_heads = 4
ff_dim = 256
num_stages = 3

# -------------------------------
# 数据集
# -------------------------------
train_dataset = PathologicalSpeechDataset(split='train')  # 训练集
val_dataset = PathologicalSpeechDataset(split='val')      # 验证集

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# -------------------------------
# 模型选择（可切换 MB-VAE / CWAFE / FAEN）
# -------------------------------
feature_extractor_type = 'MB_VAE'  # 'CWAFE' 或 'FAEN'

if feature_extractor_type == 'MB_VAE':
    feature_extractor = MBVAE.MB_VAE(embed_dim=embed_dim).to(device)
elif feature_extractor_type == 'CWAFE':
    feature_extractor = CWAFE.CWAFE(embed_dim=embed_dim).to(device)
elif feature_extractor_type == 'FAEN':
    feature_extractor = FAEN.FAEN(embed_dim=embed_dim).to(device)
else:
    raise ValueError("Unknown feature extractor type!")

# MSAT 融合模块
msat = MSAT.MSAT(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, num_stages=num_stages).to(device)

# 分类器
classifier = nn.Linear(embed_dim, num_classes).to(device)

# -------------------------------
# 损失函数和优化器
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(msat.parameters()) + list(classifier.parameters()), lr=lr)

# -------------------------------
# 训练函数
# -------------------------------
def train_one_epoch():
    feature_extractor.train()
    msat.train()
    classifier.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in train_loader:
        inputs, labels = batch  # inputs: (batch, seq_len, ...)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # -------------------------------
        # 特征提取
        # -------------------------------
        feats = feature_extractor(inputs)  # 假设输出 list [low, mid, high] 或单一 tensor
        if isinstance(feats, torch.Tensor):
            # 如果是单一特征，复制三份作为多尺度输入
            feats = [feats, feats, feats]

        # -------------------------------
        # MSAT 特征融合
        # -------------------------------
        fused_feats = msat(feats)  # (batch, seq_len, embed_dim)
        pooled = fused_feats.mean(dim=1)  # 全局平均池化 -> (batch, embed_dim)

        # -------------------------------
        # 分类器
        # -------------------------------
        outputs = classifier(pooled)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# -------------------------------
# 验证函数
# -------------------------------
def validate():
    feature_extractor.eval()
    msat.eval()
    classifier.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            feats = feature_extractor(inputs)
            if isinstance(feats, torch.Tensor):
                feats = [feats, feats, feats]

            fused_feats = msat(feats)
            pooled = fused_feats.mean(dim=1)
            outputs = classifier(pooled)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(val_dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# -------------------------------
# 训练主循环
# -------------------------------
best_val_acc = 0.0
save_path = f'msat_{feature_extractor_type}.pt'

for epoch in range(1, num_epochs+1):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()

    print(f"Epoch [{epoch}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'feature_extractor': feature_extractor.state_dict(),
            'msat': msat.state_dict(),
            'classifier': classifier.state_dict()
        }, save_path)
        print(f"Saved best model to {save_path} with Val Acc: {best_val_acc:.4f}")
