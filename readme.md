# 基于多阶段Transformer特征融合及模糊注意力机制的病理语音多分类方法研究

本项目实现了基于深度特征提取、Transformer 多阶段融合以及多分类策略的病理语音分类，支持 **MB-VAE / CWAFE / FAEN** 特征提取器，**MSAT** 特征融合模块，以及 **FLIHC / MTJC** 分类器组合。训练完成后可生成混淆矩阵、损失曲线、准确率曲线，并支持测试集推理输出预测结果。

---

## 1. 环境配置

建议使用 **Python 3.8+**，GPU环境优先。  

### 1.1 安装依赖

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn tensorboard librosa
```

### 1.2 TensorBoard

用于训练过程可视化：

```bash
tensorboard --logdir ./runs
```

---

## 2. 数据准备

1. 收集病理语音数据，**WAV 格式**。
2. 按类别建立 CSV 文件：
   - `train_labels.csv`
   - `val_labels.csv`
   - `test_labels.csv`  
   每个文件包含两列：`filename,label`。

3. 数据划分参考表 3.5-1：

| 类别编号 | 类别名称 | 样本数 | 备注 |
|----------|----------|--------|------|
| 0        | 正常语音 | 393    | 正常声带 |
| 1        | 声带结节 | 187    | 良性病变 |
| 2        | 声带息肉 | 181    | 良性病变 |
| 3        | 声带麻痹 | 178    | 功能障碍 |
| 4        | 声带水肿 | 145    | 病变 |

---

## 3. 特征提取

将 WAV 数据转换为三类特征，存入对应文件夹：

1. **频谱图（STFT）**  
2. **声学序列特征（Acoustic）**  
3. **梅尔谱图（Mel）**  

使用 `data_bz/dataset_Conversion.py` 进行转换，示例：

```python
from data_bz.dataset_Conversion import PathologicalSpeechDataset

PathologicalSpeechDataset(
    wav_dir='./wav_data',
    save_dir='./features',
    feature_types=['stft','acoustic','mel']
)
```

---

## 4. 数据加载

使用 `PathologicalSpeechDataset` 数据集类：

```python
from data_bz.dataset_Conversion import PathologicalSpeechDataset

dataset = PathologicalSpeechDataset(
    data_dir='./features',
    labels_file='./train_labels.csv',
    feature_type='mel'  # 可选 'stft', 'acoustic', 'mel'
)
```

---

## 5. 模型结构

### 5.1 特征提取器
- **MB-VAE**
- **CWAFE**
- **FAEN**

### 5.2 特征融合模块
- **MSAT**：多阶段递归 Transformer 融合模块

### 5.3 分类器
- **FLIHC**：基于模糊逻辑集成的混合分类器
- **MTJC**：多任务联合分类器

### 5.4 完整模型组合
特征提取 → MSAT 融合 → 分类器

---

## 6. 训练

完整训练脚本：`train_all.py`  

### 6.1 配置参数

```python
config = {
    'device': 'cuda',          # GPU 优先
    'batch_size': 16,
    'lr': 1e-3,
    'epochs': 100,
    'feature_type': 'mel',     # 'stft', 'acoustic', 'mel'
    'feature_extractor': 'cwafe', # 'mbvae', 'cwafe', 'faen'
    'classifier': 'flihc',     # 'flihc', 'mtjc'
    'num_classes': 5,
    'patience': 10,            # 早停
    'log_dir': './runs'
}
```

### 6.2 特性
- **早停**：防止过拟合  
- **学习率调度**：`ReduceLROnPlateau`  
- **TensorBoard 可视化**：训练损失、验证损失、准确率

### 6.3 输出
- 模型权重：`./weight/best_model.pth`  
- TensorBoard 日志：`./runs`  
- 绘图：
  - 混淆矩阵
  - 损失曲线
  - 准确率曲线

### 6.4 训练命令

```bash
python train_fusion.py
```

---

## 7. 测试推理

测试脚本：`test_model.py`

### 7.1 配置参数

```python
config = {
    'device': 'cuda',
    'batch_size': 16,
    'feature_type': 'mel',
    'feature_extractor': 'cwafe',
    'classifier': 'flihc',
    'num_classes': 5,
    'weight_path': './weight/best_model.pth'
}
```

### 7.2 功能
- 输出测试集 **准确率**
- 绘制 **混淆矩阵**
- 保存预测结果 `test_predictions.csv`

### 7.3 运行命令

```bash
python test_model.py
```

---

## 8. 文件结构示例

```
project/
│
├─ data_bz/
│   └─ dataset_Conversion.py
├─ model/
│   ├─ MB_VAE.py
│   ├─ CWAFE.py
│   └─ FAEN.py
├─ Class_tools/
│   ├─ FLIHC.py
│   └─ MTJC.py
├─ MSAT.py
├─ train_fusion.py
├─ test_model.py
├─ features/          # 存放三类特征
│   ├─ stft/
│   ├─ acoustic/
│   └─ mel/
├─ weight/            # 保存训练权重
└─ runs/              # TensorBoard 日志
```

---

## 9. 使用说明

1. 准备 WAV 数据及标签 CSV。  
2. 运行特征提取脚本生成特征文件。  
3. 修改 `train_fusion.py` 配置参数，选择特征类型、特征提取器和分类器。  
4. 训练模型。  
5. 使用 `test_model.py` 进行测试推理。  
6. 使用 TensorBoard 查看训练过程。

---

## 10. 参考

- Liu et al., **多阶段递归融合 Transformer 模块设计**  
- 模糊逻辑集成分类器（FLIHC）  
- 多任务联合分类（MTJC）  
