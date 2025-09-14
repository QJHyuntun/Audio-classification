import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# 多头注意力 + 前馈模块
# -------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (seq_len, batch, embed_dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# -------------------------------
# 阶段内多尺度融合模块（MFM）
# -------------------------------
class MultiScaleFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # 用一个TransformerBlock处理拼接的多尺度特征
        self.fusion_block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)

    def forward(self, feats):
        """
        feats: list of tensors [low, mid, high], each (batch, seq_len, embed_dim)
        """
        # 拼接频段维度: (batch, seq_len, embed_dim) -> (seq_len, batch, embed_dim)
        x = torch.stack(feats, dim=0).mean(dim=0)  # 简单均值融合，可替换为更复杂策略
        x = x.transpose(0, 1)  # 转为 (seq_len, batch, embed_dim)
        out = self.fusion_block(x)
        return out.transpose(0, 1)  # (batch, seq_len, embed_dim)

# -------------------------------
# 上下文残差增强模块（CRE）
# -------------------------------
class ContextResidualEnhance(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x, residual):
        """
        x: 当前阶段特征 (batch, seq_len, embed_dim)
        residual: 前阶段或低层残差特征 (batch, seq_len, embed_dim)
        """
        combined = torch.cat([x, residual], dim=-1)
        g = self.gate(combined)
        out = x + g * residual
        return out

# -------------------------------
# MSAT 主体结构
# -------------------------------
class MSAT(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, num_stages=3, dropout=0.1):
        super().__init__()
        self.num_stages = num_stages
        self.mfms = nn.ModuleList([MultiScaleFusion(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_stages)])
        self.cres = nn.ModuleList([ContextResidualEnhance(embed_dim) for _ in range(num_stages)])
        self.position_encoding = nn.Parameter(torch.randn(1, 1000, embed_dim))  # 最大序列长度1000，可调整

    def forward(self, feats):
        """
        feats: list of tensors [low, mid, high], each (batch, seq_len, embed_dim)
        """
        stage_input = feats
        residual = torch.stack(feats, dim=0).mean(dim=0)  # 初始残差特征

        for stage in range(self.num_stages):
            # 位置编码添加
            seq_len = stage_input[0].shape[1]
            pos_enc = self.position_encoding[:, :seq_len, :]
            stage_feats = [f + pos_enc for f in stage_input]

            # 阶段内多尺度融合
            fused = self.mfms[stage](stage_feats)

            # 残差增强
            fused = self.cres[stage](fused, residual)

            # 为下一阶段准备
            stage_input = [fused, fused, fused]  # 简单复制，可根据需求设计
            residual = fused

        return fused  # (batch, seq_len, embed_dim)
