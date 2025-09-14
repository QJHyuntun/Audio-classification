import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch

class PathologicalSpeechDataset(Dataset):
    """
    病理语音特征数据集
    可切换特征类型: 'stft', 'acoustic', 'mel'
    """
    def __init__(self, data_dir, labels_file, feature_type='stft', scaler=None):
        """
        Args:
            data_dir (str): 特征文件所在根目录，包含 'spectrogram', 'acoustic_seq', 'mel_spectrogram'
            labels_file (str): 标签文件路径，格式: filename,label
            feature_type (str): 特征类型, 'stft' / 'acoustic' / 'mel'
            scaler (sklearn.preprocessing.StandardScaler, optional): 是否使用标准化
        """
        assert feature_type in ['stft', 'acoustic', 'mel'], "feature_type 必须为 'stft', 'acoustic', 'mel'"
        self.data_dir = data_dir
        self.feature_type = feature_type
        self.scaler = scaler

        # 加载标签文件
        self.samples = []
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    fname, label = line.split(',')
                    self.samples.append((fname, int(label)))

        # 对应特征文件夹
        self.feature_folder = os.path.join(data_dir, {
            'stft': 'spectrogram',
            'acoustic': 'acoustic_seq',
            'mel': 'mel_spectrogram'
        }[feature_type])

        # 构建文件路径
        self.feature_paths = []
        for fname, _ in self.samples:
            feature_path = os.path.join(self.feature_folder, f"{fname}_{feature_type}.npy")
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"特征文件不存在: {feature_path}")
            self.feature_paths.append(feature_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        X = np.load(feature_path)  # 加载特征
        y = self.samples[idx][1]   # 标签

        # 标准化
        if self.scaler:
            shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1]) if X.ndim > 1 else X.reshape(-1, 1)
            X_scaled = self.scaler.transform(X_flat)
            X = X_scaled.reshape(shape)

        # 转为 tensor
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        # 对于声学序列或 STFT/Mel 特征，如果需要 channel first，可根据模型调整
        if X.ndim == 2:
            X = X.unsqueeze(0)  # (1, F, T)

        return X, y

if __name__ == '__main__':
    dataset = PathologicalSpeechDataset(
        data_dir='./features',
        labels_file='./labels.csv',
        feature_type='mel',
        scaler=None
    )
    print(len(dataset))
    X, y = dataset[0]
    print(X.shape, y)
