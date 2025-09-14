import os
import torch
from torch.utils.data import DataLoader
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
    'feature_type': 'mel',           # 'stft', 'acoustic', 'mel'
    'feature_extractor': 'cwafe',    # 'mbvae', 'cwafe', 'faen'
    'classifier': 'flihc',           # 'flihc', 'mtjc'
    'num_classes': 5,
    'weight_path': './weight/best_model.pth'
}

# ====== 数据集 ======
test_dataset = PathologicalSpeechDataset(
    data_dir='./features',
    labels_file='./test_labels.csv',
    feature_type=config['feature_type']
)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

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
class FullModel(torch.nn.Module):
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

# ====== 加载权重 ======
model.load_state_dict(torch.load(config['weight_path'], map_location=config['device']))
model.eval()

# ====== 推理 ======
all_preds, all_labels = [], []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(config['device']), y.to(config['device'])
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# ====== 准确率 ======
acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc:.4f}")

# ====== 混淆矩阵 ======
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Confusion Matrix')
plt.show()

# ====== 保存预测结果 ======
import pandas as pd
results_df = pd.DataFrame({
    'TrueLabel': all_labels,
    'Predicted': all_preds
})
results_df.to_csv('test_predictions.csv', index=False)
print("Predictions saved to test_predictions.csv")
