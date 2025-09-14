import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


# ----------------------
#  Conformer 模块
# ----------------------
class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm)
        return out


class ConvolutionModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.silu = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B, T, C]
        return x


class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, expansion_factor=4, dropout=0.1, kernel_size=31):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, expansion_factor, dropout)
        self.mhsa = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.conv = ConvolutionModule(dim, kernel_size, dropout)
        self.ffn2 = FeedForwardModule(dim, expansion_factor, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


# ----------------------
#  CWAFE 模型
# ----------------------
class CWAFE(nn.Module):
    def __init__(self, num_classes=5, num_conformer_blocks=2, d_model=768, num_heads=8):
        super().__init__()
        # Wav2Vec2 预训练模型
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        # Conformer Block 堆叠
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads=num_heads) for _ in range(num_conformer_blocks)
        ])

        # 线性分类器
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_values, attention_mask=None):
        # 1. Wav2Vec2 提取特征
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # [B, T, D]

        # 2. Conformer Blocks
        for block in self.conformer_blocks:
            x = block(x)

        # 3. 池化 (取时序平均池化作为全局特征)
        x = x.mean(dim=1)  # [B, D]

        # 4. 分类
        logits = self.classifier(x)
        return logits

