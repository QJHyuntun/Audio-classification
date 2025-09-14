import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 基础模块：单频段 VAE Encoder ==========
class BandVAEEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        """
        input_shape: (channels, freq_bins, time_frames)
        latent_dim: 潜变量维度
        """
        super(BandVAEEncoder, self).__init__()
        c, f, t = input_shape

        # 卷积特征提取
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),  # (32, f, t)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 下采样 (64, f/2, t/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 计算 flatten 后的维度
        with torch.no_grad():
            dummy = torch.zeros(1, c, f, t)
            out = self.conv(dummy)
            flat_dim = out.numel()

        # 潜变量均值和 log 方差
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)   # 展平
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # 重参数化
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


# ========== MB-VAE 总体模型 ==========
class MBVAE(nn.Module):
    def __init__(self, band_shapes, latent_dim=32, num_classes=5, fusion="concat"):
        """
        band_shapes: [(c, f1, t), (c, f2, t), (c, f3, t)] 三个频段的输入尺寸
        latent_dim: 每个频段的潜变量维度
        num_classes: 分类类别数
        fusion: 'concat' 或 'weighted'
        """
        super(MBVAE, self).__init__()
        self.fusion = fusion
        self.encoders = nn.ModuleList([
            BandVAEEncoder(shape, latent_dim) for shape in band_shapes
        ])

        if fusion == "concat":
            fused_dim = latent_dim * 3
        elif fusion == "weighted":
            fused_dim = latent_dim
            self.weights = nn.Parameter(torch.ones(3))  # 可学习权重
        else:
            raise ValueError("fusion must be 'concat' or 'weighted'")

        # 基础线性分类器
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, num_classes)
        )

    def forward(self, bands):
        """
        bands: [low_band, mid_band, high_band]
        每个 band 输入 shape = (B, C, F, T)
        """
        zs, mus, logvars = [], [], []
        for i, enc in enumerate(self.encoders):
            z, mu, logvar = enc(bands[i])
            zs.append(z)
            mus.append(mu)
            logvars.append(logvar)

        if self.fusion == "concat":
            z_fused = torch.cat(zs, dim=1)
        elif self.fusion == "weighted":
            weights = F.softmax(self.weights, dim=0)
            z_fused = sum(w * z for w, z in zip(weights, zs))

        logits = self.classifier(z_fused)
        return logits, zs, mus, logvars


# ========== 损失函数 ==========
def mbvae_loss(logits, labels, mus, logvars, alpha_kl=0.1):
    """
    logits: 分类输出
    labels: 真实标签
    mus, logvars: 每个频段的均值与方差
    alpha_kl: KL loss 权重
    """
    # 分类损失
    cls_loss = F.cross_entropy(logits, labels)

    # KL 损失
    kl_loss = 0
    for mu, logvar in zip(mus, logvars):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss += kl.mean()
    kl_loss = kl_loss / len(mus)

    return cls_loss + alpha_kl * kl_loss, cls_loss, kl_loss
