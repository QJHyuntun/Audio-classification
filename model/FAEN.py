"""
FAEN - Fuzzy Attention Enhanced Network for mel-spectrogram feature extraction
Backbone: MobileNetV2 (torchvision)
Fuzzy Attention Modules (FAM) placed after backbone.features[3], [5], [7]
Classifier: simple linear layer on pooled fused features

Author: (generated)
Requirements: torch, torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


# -----------------------------
# Utility: sliding window local variance & gradient (Sobel-like)
# -----------------------------
def local_variance_map(x, kernel_size=7, eps=1e-6):
    """
    x: (B,1,H,W) - single-channel spectrogram normalized to [0,1] (or any range)
    returns: (B,1,H,W) local variance estimate via conv
    """
    # mean filter
    padding = kernel_size // 2
    weight = torch.ones(1, 1, kernel_size, kernel_size, device=x.device) / (kernel_size * kernel_size)
    mean = F.conv2d(x, weight, padding=padding)
    mean_sq = F.conv2d(x * x, weight, padding=padding)
    var = mean_sq - mean * mean
    var = torch.clamp(var, min=0.0)
    return var


def gradient_magnitude_map(x):
    """
    approximate gradient magnitude using Sobel kernels
    x: (B,1,H,W)
    returns: (B,1,H,W)
    """
    # Sobel kernels (for single channel); use 3x3
    kernel_x = torch.tensor([[[-1., 0., 1.],
                              [-2., 0., 2.],
                              [-1., 0., 1.]]], device=x.device).unsqueeze(0)
    kernel_y = torch.tensor([[[-1., -2., -1.],
                              [ 0.,  0.,  0.],
                              [ 1.,  2.,  1.]]], device=x.device).unsqueeze(0)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    gm = torch.sqrt(gx * gx + gy * gy + 1e-8)
    return gm


# -----------------------------
# Fuzzy Attention Module (FAM)
# -----------------------------
class FuzzyAttentionModule(nn.Module):
    def __init__(self, in_channels, inter_channels=None, kernel_var=7, spatial_conv_channels=32):
        """
        in_channels: channels of feature map at this insertion point
        inter_channels: hidden channels for MLP in channel-attention
        kernel_var: sliding window size for local variance
        """
        super().__init__()
        if inter_channels is None:
            inter_channels = max(8, in_channels // 8)

        self.kernel_var = kernel_var

        # Channel attention MLP
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels * 3, inter_channels),
            nn.ReLU(),
            nn.Linear(inter_channels, in_channels),
            nn.Sigmoid()
        )

        # Spatial attention convs (process PHFS maps concatenated)
        # input channels for spatial conv = 3 (mu, nu, pi)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(3, spatial_conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_conv_channels),
            nn.ReLU(),
            nn.Conv2d(spatial_conv_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # small projection to match feature scale if needed
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    @staticmethod
    def phfs_from_score(s):
        """
        s: (B,1,H,W) fuzziness score in [0,1] (higher => more fuzzy/uncertain)
        Return mu, nu, pi such that mu^2 + nu^2 <= 1 and pi = sqrt(1 - mu^2 - nu^2)
        Design approach:
            - raw_mu = 1 - s  (more fuzzy -> lower membership)
            - raw_nu = s      (more fuzzy -> higher non-membership)
            - scale factor r = min(1, 1 / sqrt(raw_mu^2 + raw_nu^2 + eps))
            - mu = raw_mu * r, nu = raw_nu * r
            - pi = sqrt(max(0, 1 - mu^2 - nu^2))
        This ensures numerical safety and PHFS constraint.
        """
        eps = 1e-8
        raw_mu = 1.0 - s
        raw_nu = s
        denom = torch.sqrt(raw_mu * raw_mu + raw_nu * raw_nu + eps)
        r = torch.clamp(1.0 / (denom + eps), max=1.0)
        mu = raw_mu * r
        nu = raw_nu * r
        tmp = 1.0 - mu * mu - nu * nu
        tmp = torch.clamp(tmp, min=0.0)
        pi = torch.sqrt(tmp + eps)
        return mu, nu, pi

    def forward(self, feat, mel_spec):
        """
        feat: (B, C, Hf, Wf) feature map from backbone at this stage
        mel_spec: (B, 1, Hs, Ws) original mel-spectrogram (single channel)
        Steps:
            - downsample mel_spec -> spatial size of feat (Hf, Wf)
            - compute local variance & gradient maps
            - compute fuzziness score S (combine normalized var & grad)
            - compute PHFS triple (mu, nu, pi)
            - channel attention: global pool each of mu/nu/pi over spatial -> concat -> MLP -> channel weights
            - spatial attention: conv over stacked (mu,nu,pi) -> spatial map
            - fused attention = channel_map (broadcast) * spatial_map
            - apply to feat: out = feat * (1 + fused_att)
        """
        B, C, Hf, Wf = feat.shape
        # 1) Resize mel_spec to match feat spatial dims
        spec_resized = F.interpolate(mel_spec, size=(Hf, Wf), mode='bilinear', align_corners=False)  # (B,1,Hf,Wf)

        # 2) local variance and gradient
        var_map = local_variance_map(spec_resized, kernel_size=self.kernel_var)  # (B,1,Hf,Wf)
        grad_map = gradient_magnitude_map(spec_resized)  # (B,1,Hf,Wf)

        # Normalize maps to [0,1] per-sample for stability
        def min_max_norm(x):
            x_min = x.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
            x_max = x.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
            return (x - x_min) / (x_max - x_min + 1e-8)

        var_n = min_max_norm(var_map)
        grad_n = min_max_norm(grad_map)

        # 3) fuzziness score S: combine var & grad (weighted sum)
        S = (var_n + grad_n) / 2.0  # (B,1,Hf,Wf), in [0,1]

        # 4) PHFS triple
        mu, nu, pi = self.phfs_from_score(S)  # each (B,1,Hf,Wf)

        # 5) Channel attention: global pooling over spatial for mu,nu,pi -> concat -> MLP
        # Global average pool
        mu_gap = mu.view(B, -1).mean(dim=1)  # (B,)
        nu_gap = nu.view(B, -1).mean(dim=1)
        pi_gap = pi.view(B, -1).mean(dim=1)
        channel_input = torch.cat([mu_gap, nu_gap, pi_gap], dim=1) if False else torch.stack([mu_gap, nu_gap, pi_gap], dim=1).view(B, -1)
        # channel_input shape (B, 3)
        # expand to in_channels*3 for MLP input compatibility:
        channel_input_expanded = channel_input.repeat(1, C)  # (B, 3*C)  -> alternative: project directly
        # More stable/compact: create single vector [mu_gap,nu_gap,pi_gap] then transform to C dims:
        # use a small MLP that expects input dim 3 and outputs C
        ch_mlp = nn.Sequential(
            nn.Linear(3, max(8, C // 8)),
            nn.ReLU(),
            nn.Linear(max(8, C // 8), C),
            nn.Sigmoid()
        ).to(feat.device)
        # compute channel weights
        channel_weights = ch_mlp(channel_input)  # (B, C)
        channel_weights = channel_weights.view(B, C, 1, 1)  # broadcastable

        # 6) Spatial attention from stacked PHFS maps
        phfs_stack = torch.cat([mu, nu, pi], dim=1)  # (B,3,Hf,Wf)
        spatial_map = self.spatial_conv(phfs_stack)  # (B,1,Hf,Wf) in (0,1)

        # 7) fused attention
        fused_att = channel_weights * spatial_map  # (B,C,Hf,Wf) broadcast multiplication

        # small projection (optional) so feature scale preserved
        feat_proj = self.proj(feat)

        # 8) residual enhancement
        out = feat_proj * (1.0 + fused_att)
        return out, (mu, nu, pi, S)


# -----------------------------
# FAEN model (MobileNetV2 + 3xFAM + classifier)
# -----------------------------
class FAEN(nn.Module):
    def __init__(self, num_classes=5, pretrained_backbone=True):
        super().__init__()
        # load mobilenet_v2 backbone
        backbone = mobilenet_v2(pretrained=pretrained_backbone)
        features = backbone.features  # nn.Sequential of InvertedResidual blocks + initial conv

        # We'll use features up to the last block (exclude classifier)
        self.features = features

        # Determine insertion points: choose indices 3,5,7 (0-based in features)
        # Note: torchvision mobilenet_v2 features layout typical length = 18.
        self.insert_indices = [3, 5, 7]

        # Create FAM modules for each insertion point
        # we need to know channel dims at those points; map empirically from mobilenet_v2 default:
        # but safer: we will fetch channel dims dynamically on first forward via lazy init.
        self.fams = nn.ModuleList([None for _ in self.insert_indices])  # placeholders

        # final projection after backbone to produce base feature
        # We'll lazily create a conv to project final feature to a canonical dim if needed
        self.final_proj = None

        # classifier (will be configured after we know final fused channel dim)
        self.classifier = None

        # store whether lazy init done
        self._lazy_init_done = False

        # store backbone for forward slicing convenience
        self._backbone = backbone

    def _lazy_init(self, x_spec, device):
        """
        Execute a dummy forward through features to determine channel dims and spatial sizes,
        then create FAMs and classifier accordingly. x_spec is (B,1,Hs,Ws) mel spectrogram.
        """
        B = 1
        dummy = torch.zeros(B, 1, x_spec.shape[-2], x_spec.shape[-1], device=device)  # (1,1,Hs,Ws)
        # convert spec into a fake "image" with channel 1 and time as width; but backbone expects 3-channel images
        # We'll create dummy image by repeating channel to 3-ch and resizing to typical input (e.g., 224x224)
        # Instead, we forward a dummy image of shape (1,3,224,224) through features to gather shapes:
        img = torch.zeros(1, 3, 224, 224, device=device)
        feats = []
        x = img
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.insert_indices:
                feats.append(x)
        final_feat = x
        # now create FAM modules with channel dims from feats
        for i, f in enumerate(feats):
            ch = f.shape[1]
            self.fams[i] = FuzzyAttentionModule(in_channels=ch)
        # final projection: project final_feat channels to a canonical dim (sum of fam outputs channels after concat)
        fused_ch = final_feat.shape[1] + sum([f.shape[1] for f in feats])
        # set final projection if needed (identity if dims already OK)
        self.final_proj = nn.Conv2d(fused_ch, fused_ch, kernel_size=1)
        # classifier: global pooled -> linear
        self.classifier = nn.Linear(fused_ch,  num_classes := 5)  # default 5, but user can adjust after init if needed
        self._lazy_init_done = True

    def forward(self, mel_spec):
        """
        mel_spec: (B,1,Hs,Ws) single-channel mel-spectrogram, e.g., shape (B,1,128,256)
        returns: logits (B, num_classes), and optionally debug dict
        """
        B, _, Hs, Ws = mel_spec.shape
        device = mel_spec.device

        # Lazy init if necessary
        if not self._lazy_init_done:
            self._lazy_init(mel_spec, device)

        # We'll feed a 3-channel "image" to mobilenet features constructed from mel_spec by repeating channels
        # Resize mel_spec to a square-ish size expected by backbone (224x224) to get sensible feature maps
        img = F.interpolate(mel_spec, size=(224, 224), mode='bilinear', align_corners=False)
        img3 = img.repeat(1, 3, 1, 1)  # (B,3,224,224)
        feats_at_inserts = []
        fam_outputs = []

        x = img3
        fam_i = 0
        for idx, layer in enumerate(self.features):
            x = layer(x)  # propagate
            if idx in self.insert_indices:
                # x is feature at this insertion point
                # feed original mel_spec (resized to current x spatial) and feature to FAM
                Bf, Cf, Hf, Wf = x.shape
                spec_resized = F.interpolate(mel_spec, size=(Hf, Wf), mode='bilinear', align_corners=False)
                fam_module = self.fams[fam_i]
                fam_out, fam_debug = fam_module(x, spec_resized)
                fam_outputs.append(fam_out)
                feats_at_inserts.append(x)
                fam_i += 1

        final_feat = x  # backbone final feature
        # concat final_feat and fam_outputs (channel-wise)
        cat_list = [final_feat] + fam_outputs
        fused = torch.cat(cat_list, dim=1)  # (B, sumC, Hf, Wf) - Hf/Wf consistent with final_feat

        # optional projection
        fused = self.final_proj(fused)

        # global average pool
        gap = F.adaptive_avg_pool2d(fused, (1, 1)).view(B, -1)  # (B, fused_ch)

        logits = self.classifier(gap)

        # debug info: return PHFS triples optionally; but here we'll not return large maps by default
        return logits

