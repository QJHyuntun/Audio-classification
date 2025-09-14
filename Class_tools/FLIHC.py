# model/flihc.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FLIHC(nn.Module):
    """
    Fuzzy Logic Integrated Hybrid Classification (FLIHC)
    用于融合多个基础分类器输出的模糊逻辑集成模块
    """
    def __init__(self, num_classes=5, base_classifier_dims=[128,128,128], device='cuda'):
        """
        num_classes: 分类总数
        base_classifier_dims: 每个基础分类器输出特征维度列表
        """
        super(FLIHC, self).__init__()
        self.num_classes = num_classes
        self.num_classifiers = len(base_classifier_dims)
        self.device = device

        # 为每个基础分类器初始化线性输出层
        self.classifiers = nn.ModuleList([
            nn.Linear(dim, num_classes) for dim in base_classifier_dims
        ])

        # 模糊隶属度层参数，可学习
        # alpha: 控制隶属度曲线的斜率
        self.alpha = nn.Parameter(torch.ones(self.num_classifiers, 1, 1) * 1.0)

        # 分类器权重参数，用于模糊推理加权融合
        self.classifier_weights = nn.Parameter(torch.ones(self.num_classifiers, num_classes) / self.num_classifiers)

    def phfs_membership(self, logits):
        """
        将每个分类器的logits转化为模糊隶属度
        logits: list of tensor, 每个 tensor shape (batch, num_classes)
        return: tensor (batch, num_classes, num_classifiers)
        """
        batch_size = logits[0].size(0)
        membership_list = []
        for i, logit in enumerate(logits):
            prob = F.softmax(logit, dim=-1)  # 概率
            # PHFS隶属度映射: u = prob ^ alpha
            u = prob ** torch.clamp(self.alpha[i], min=0.1, max=10)
            membership_list.append(u.unsqueeze(-1))  # (batch, num_classes, 1)
        # 拼接所有分类器的隶属度
        membership_tensor = torch.cat(membership_list, dim=-1)  # (batch, num_classes, num_classifiers)
        return membership_tensor

    def fuzzy_inference(self, membership_tensor):
        """
        模糊推理层: 加权融合各分类器隶属度
        membership_tensor: (batch, num_classes, num_classifiers)
        return: (batch, num_classes)
        """
        # 权重归一化
        weights = F.softmax(self.classifier_weights, dim=0)  # (num_classifiers, num_classes)
        # 转置以便与 membership_tensor 对齐
        weights = weights.transpose(0,1).unsqueeze(0)  # (1, num_classes, num_classifiers)
        # 加权融合
        fused_membership = (membership_tensor * weights).sum(dim=-1)  # (batch, num_classes)
        return fused_membership

    def forward(self, features_list):
        """
        features_list: list of feature tensors，每个 tensor shape (batch, feature_dim)
        """
        # 1. 基础分类器输出 logits
        logits_list = [classifier(feat) for classifier, feat in zip(self.classifiers, features_list)]
        # 2. PHFS隶属度映射
        membership_tensor = self.phfs_membership(logits_list)
        # 3. 模糊推理加权融合
        fused_membership = self.fuzzy_inference(membership_tensor)
        # 4. 最大隶属度判定（可直接用于训练 softmax + CE loss）
        return fused_membership
