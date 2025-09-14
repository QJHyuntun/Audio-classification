import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_bz.dataset_Conversion import PathologicalSpeechDataset

# ====== 导入模型 ======
from MSAT import MSAT
from model.MB_VAE import MBVAE
from model.CWAFE import CWAFE
from model.FAEN import FAEN
from Class_tools.FLIHC import FLIHC
from Class_tools.MTJC import MTJC

# ====== 配置参数 ======
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 16,
    'lr': 1e-3,
    'epochs': 100,
    'feature_type': 'mel',           # 'stft', 'acoustic', 'mel'
    'feature_extractor': 'cwafe',    # 'mbvae', 'cwafe', 'faen'
    'classifier': 'flihc',           # 'flihc', 'mtjc'
    'num_classes': 5,
    'patience': 10,                  # 早停轮数
    'log_dir': './runs',
    'weight_dir': './weight'
}

# ====== 创建 weight 文件夹 ======
os.makedirs(config['weight_dir'], exist_ok=True)

# ====== 数据集 ======
train_dataset = PathologicalSpeechDataset(
    data_dir='./features',
    labels_file='./train_labels.csv',
    feature_type=config['feature_type']
)
val_dataset = PathologicalSpeechDataset(
    data_dir='./features',
    labels_file='./val_labels.csv',
    feature_type=config['feature_type']
)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

# ====== 初始化特征提取器 ======
if config['feature_extractor'] == 'mbvae':
    feature_extractor = MBVAE()
elif config['feature_extractor'] == 'cwafe':
    feature_extractor = CWAFE()
elif config['feature_extractor'] == 'faen':
    feature_extractor = FAEN()
else:
    raise ValueError("Unknown feature_extractor")

# ====== 特征融合模块 ======
fusion_module = MSAT(in_channels=feature_extractor.output_dim, hidden_dim=256)

# ====== 分类器 ======
if config['classifier'] == 'flihc':
    classifier = FLIHC(input_dim=256, num_classes=config['num_classes'])
elif config['classifier'] == 'mtjc':
    classifier = MTJC(input_dim=256, num_classes=config['num_classes'], aux_tasks=2)
else:
    raise ValueError("Unknown classifier")

# ====== 合并模型 ======
class FullModel(nn.Module):
    def __init__(self, feature_extractor, fusion_module, classifier):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fusion_module = fusion_module
        self.classifier = classifier

    def forward(self, x):
        feat = self.feature_extractor(x)
        fused = self.fusion_module(feat)
        out = self.classifier(fused)
        return out

model = FullModel(feature_extractor, fusion_module, classifier).to(config['device'])

# ====== 损失函数与优化器 ======
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=config['lr'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# ====== TensorBoard ======
writer = SummaryWriter(config['log_dir'])

# ====== 训练 ======
train_losses, val_losses, val_accuracies = [], [], []
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(config['epochs']):
    model.train()
    running_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(config['device']), y.to(config['device'])
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # ====== 验证 ======
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(config['device']), y.to(config['device'])
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item() * X.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_acc = accuracy_score(all_labels, all_preds)
    val_accuracies.append(val_acc)

    # ====== 调度学习率 ======
    scheduler.step(val_loss)

    # ====== TensorBoard日志 ======
    writer.add_scalar('Loss/train', train_loss, epoch+1)
    writer.add_scalar('Loss/val', val_loss, epoch+1)
    writer.add_scalar('Accuracy/val', val_acc, epoch+1)

    print(f"Epoch [{epoch+1}/{config['epochs']}], "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # ====== 早停 ======
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), os.path.join(config['weight_dir'], 'best_model.pth'))
    else:
        early_stop_counter += 1
        if early_stop_counter >= config['patience']:
            print("Early stopping triggered")
            break

# ====== 绘制混淆矩阵 ======
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ====== 绘制损失曲线 ======
plt.figure()
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.show()

# ====== 绘制准确率曲线 ======
plt.figure()
plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy Curve')
plt.show()

writer.close()
