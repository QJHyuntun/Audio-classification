# model/hierarchical_msat.py
import torch
import torch.nn as nn
import math

class PositionalEncoding2D(nn.Module):
    """
    2D 空间结构增强位置编码
    """
    def __init__(self, embed_dim, height, width):
        super(PositionalEncoding2D, self).__init__()
        self.embed_dim = embed_dim
        self.height = height
        self.width = width

        pe = torch.zeros(embed_dim, height, width)
        if embed_dim % 4 != 0:
            raise ValueError("Embed_dim must be divisible by 4 for 2D PE.")
        d_model = embed_dim // 2
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0,1).unsqueeze(1).repeat(1,height,1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0,1).unsqueeze(1).repeat(1,height,1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0,1).unsqueeze(2).repeat(1,1,width)
        pe[d_model+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0,1).unsqueeze(2).repeat(1,1,width)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, C, H, W)

    def forward(self, x):
        """
        x: (batch, C, H, W)
        """
        return x + self.pe


class HierarchicalMSAT(nn.Module):
    """
    多尺度分层融合Transformer模块
    """
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, num_layers=2, num_scales=3):
        super(HierarchicalMSAT, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.num_scales = num_scales

        # 为每个尺度初始化一个Transformer Encoder
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    batch_first=True
                ),
                num_layers=num_layers
            ) for _ in range(num_scales)
        ])

        # 动态权重调节 MLP
        self.weight_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_scales)
        ])

        # 可学习的尺度嵌入
        self.scale_embeddings = nn.Parameter(torch.randn(num_scales, embed_dim))

    def forward(self, feats):
        """
        feats: list of length num_scales, 每个元素 shape: (batch, seq_len, embed_dim)
        """
        batch_size = feats[0].size(0)
        seq_len = feats[0].size(1)

        # 分层递进融合
        fused = None
        outputs = []
        for i, feat in enumerate(feats):
            # 加入尺度嵌入
            feat = feat + self.scale_embeddings[i].unsqueeze(0).unsqueeze(0)  # (1,1,embed_dim)

            # 与上一尺度融合（递进拼接+线性映射）
            if fused is not None:
                feat = torch.cat([fused, feat], dim=-1)  # (batch, seq_len, 2*embed_dim)
                # 投影回 embed_dim
                feat = nn.Linear(feat.size(-1), self.embed_dim).to(feat.device)(feat)

            # Transformer编码
            feat = self.transformers[i](feat)  # (batch, seq_len, embed_dim)

            # 动态权重调节
            w = self.weight_mlps[i](feat.mean(dim=1))  # (batch,1)
            feat = feat * w.unsqueeze(1)

            fused = feat
            outputs.append(fused)

        # 最终融合结果（加权求和）
        final_feat = torch.stack(outputs, dim=0).sum(dim=0)  # (batch, seq_len, embed_dim)
        return final_feat
