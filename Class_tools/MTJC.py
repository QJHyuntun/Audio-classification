# model/mtjc.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MTJC(nn.Module):
    """
    Multi-Task Joint Classification (MTJC)
    支持主任务多类别分类 + 多辅助任务（回归/二分类）
    """
    def __init__(self, input_dim, num_classes=5, aux_tasks=None, hidden_dim=128):
        """
        input_dim: 特征维度
        num_classes: 主任务分类数量
        aux_tasks: list of dict，每个 dict 定义辅助任务 {'name':str, 'type':'regression'/'classification', 'out_dim':int}
        hidden_dim: 隐藏层维度
        """
        super(MTJC, self).__init__()
        self.num_classes = num_classes
        self.aux_tasks = aux_tasks if aux_tasks is not None else []

        # 共享编码层
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 主任务分类器
        self.main_classifier = nn.Linear(hidden_dim, num_classes)

        # 辅助任务分类器/回归器
        self.aux_classifiers = nn.ModuleDict()
        for task in self.aux_tasks:
            out_dim = task.get('out_dim', 1)
            if task['type'] == 'regression':
                self.aux_classifiers[task['name']] = nn.Linear(hidden_dim, out_dim)
            elif task['type'] == 'classification':
                self.aux_classifiers[task['name']] = nn.Linear(hidden_dim, out_dim)
            else:
                raise ValueError(f"Unsupported auxiliary task type: {task['type']}")

    def forward(self, x):
        """
        x: 输入特征, shape (batch, input_dim)
        return: dict {'main': logits, 'aux': {task_name: output}}
        """
        shared_feat = self.shared_encoder(x)
        main_out = self.main_classifier(shared_feat)

        aux_outs = {}
        for task_name, classifier in self.aux_classifiers.items():
            aux_outs[task_name] = classifier(shared_feat)
        return {'main': main_out, 'aux': aux_outs}

    def compute_loss(self, outputs, labels, aux_labels=None, aux_weights=None):
        """
        outputs: forward() 输出
        labels: 主任务标签, (batch,)
        aux_labels: dict, 每个任务对应标签 tensor
        aux_weights: dict, 每个任务对应权重
        """
        loss = F.cross_entropy(outputs['main'], labels)

        if aux_labels is not None:
            aux_weights = aux_weights if aux_weights is not None else {k:1.0 for k in aux_labels.keys()}
            for task_name, target in aux_labels.items():
                pred = outputs['aux'][task_name]
                w = aux_weights.get(task_name, 1.0)
                if len(pred.shape) == 1 or pred.shape[1] == 1:  # 回归
                    loss += w * F.mse_loss(pred.squeeze(), target.float())
                else:  # 分类
                    loss += w * F.cross_entropy(pred, target)
        return loss
