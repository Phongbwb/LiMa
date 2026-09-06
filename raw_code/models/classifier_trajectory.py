import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models

# ==========================================
# 0. KHỞI TẠO MODULE MAMBA & CFC (FALLBACKS)
# ==========================================
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
    print("⚠️ Cảnh báo: mamba_ssm chưa được cài đặt. Sẽ dùng Fallback.")

try:
    from ncps.torch import CfC
except ImportError:
    CfC = None
    print("⚠️ Cảnh báo: ncps chưa được cài đặt. Nhánh Trajectory sẽ dùng GRU thay thế.")

# ==========================================
# 1. BACKBONE: IMAGE FEATURE EXTRACTOR (cho ảnh tĩnh)
# ==========================================
class CustomVideoBackbone(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.early_extractor = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        )
        for param in self.early_extractor.parameters():
            param.requires_grad = False

        self.resnet_shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(1, d_model)
        )

        self.spatial_mlp = nn.Conv2d(18, d_model, 1)  # 18 = 2 + 2*4*2 Fourier features
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(d_model)

    def get_fourier_features(self, coords, num_bands=4):
        features = [coords]
        for freq in [2**i for i in range(num_bands)]:
            features.append(torch.sin(math.pi * freq * coords))
            features.append(torch.cos(math.pi * freq * coords))
        return torch.cat(features, dim=-1)

    def forward(self, x):
        # x: (B, C, H, W) — ảnh tĩnh
        feat_2d = self.resnet_shortcut(self.early_extractor(x))  # (B, D, Hp, Wp)
        _, E, Hp, Wp = feat_2d.shape

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, Hp, device=x.device),
            torch.linspace(-1, 1, Wp, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        grid_fourier = self.get_fourier_features(grid).permute(0, 3, 1, 2)
        spatial_pe = self.spatial_mlp(grid_fourier).permute(0, 2, 3, 1)  # (1, Hp, Wp, D)

        # Output: (B, Hp, Wp, D)
        out = feat_2d.permute(0, 2, 3, 1) + spatial_pe * self.pe_scale
        return self.norm(out), spatial_pe  # (B, H, W, D), (1, H, W, D)


# ==========================================
# 2. LÕI MAMBA 1D
# ==========================================
class MambaCore(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.x_proj = nn.Linear(d_model, d_model + 2 * d_state)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(d_model, 1)))
        self.D_param = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_br, z_br = xz.chunk(2, dim=-1)

        x_conv = F.silu(self.conv1d(x_br.transpose(1, 2))).transpose(1, 2)
        proj = self.x_proj(x_conv)
        delta, B_mat, C_mat = torch.split(proj, [D, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())

        if selective_scan_fn is None:
            return self.out_proj(x_conv * F.silu(z_br))

        y = selective_scan_fn(
            x_conv.transpose(1, 2).contiguous(),
            delta.transpose(1, 2).contiguous(),
            A,
            B_mat.transpose(1, 2).contiguous(),
            C_mat.transpose(1, 2).contiguous(),
            self.D_param.float(),
            delta_softplus=True
        ).transpose(1, 2)

        return self.out_proj(y * F.silu(z_br))


# ==========================================
# 3. SPATIAL MAMBA BLOCK (cho ảnh 2D, không có T)
# ==========================================
class SpatialMambaBlock(nn.Module):
    """Quét 4 hướng trên ảnh 2D (B, H, W, D)"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.spatial_conv2d = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.core_h_fwd = MambaCore(d_model, d_state)
        self.core_h_bwd = MambaCore(d_model, d_state)
        self.core_v_fwd = MambaCore(d_model, d_state)
        self.core_v_bwd = MambaCore(d_model, d_state)

    def forward(self, x_norm):
        # x_norm: (B, H, W, D)
        B, H, W, D = x_norm.shape

        x_2d = x_norm.permute(0, 3, 1, 2).contiguous()         # (B, D, H, W)
        x_conv = x_2d + self.spatial_conv2d(x_2d)               # (B, D, H, W)

        # Quét ngang: (B, H*W, D)
        x_h = x_conv.permute(0, 2, 3, 1).contiguous().view(B, H * W, D)
        h_fwd = self.core_h_fwd(x_h)
        h_bwd = self.core_h_bwd(x_h.flip(dims=[1])).flip(dims=[1])

        # Quét dọc: (B, W*H, D) -> transpose lại
        x_v = x_conv.permute(0, 3, 2, 1).contiguous().view(B, W * H, D)
        v_fwd_out = self.core_v_fwd(x_v)
        v_bwd_out = self.core_v_bwd(x_v.flip(dims=[1])).flip(dims=[1])

        v_fwd = v_fwd_out.view(B, W, H, D).transpose(1, 2).contiguous().view(B, H * W, D)
        v_bwd = v_bwd_out.view(B, W, H, D).transpose(1, 2).contiguous().view(B, H * W, D)

        out = (h_fwd + h_bwd + v_fwd + v_bwd) / 4.0
        return out.view(B, H, W, D)


class ContextMambaBlock(nn.Module):
    """Quét toàn bộ feature map 4 hướng (B, H, W, D)"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.spatial_conv2d = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.core_h_fwd = MambaCore(d_model, d_state)
        self.core_h_bwd = MambaCore(d_model, d_state)
        self.core_v_fwd = MambaCore(d_model, d_state)
        self.core_v_bwd = MambaCore(d_model, d_state)

    def forward(self, x_norm):
        # x_norm: (B, H, W, D)
        B, H, W, D = x_norm.shape

        x_2d = x_norm.permute(0, 3, 1, 2).contiguous()       # (B, D, H, W)
        x_conv = x_2d + self.spatial_conv2d(x_2d)             # (B, D, H, W)

        # Quét ngang
        x_h = x_conv.permute(0, 2, 3, 1).contiguous().view(B, H * W, D)
        h_fwd = self.core_h_fwd(x_h)
        h_bwd = self.core_h_bwd(x_h.flip(dims=[1])).flip(dims=[1])

        # Quét dọc
        x_v = x_conv.permute(0, 3, 2, 1).contiguous().view(B, W * H, D)
        v_fwd_out = self.core_v_fwd(x_v)
        v_bwd_out = self.core_v_bwd(x_v.flip(dims=[1])).flip(dims=[1])

        v_fwd = v_fwd_out.view(B, W, H, D).transpose(1, 2).contiguous().view(B, H * W, D)
        v_bwd = v_bwd_out.view(B, W, H, D).transpose(1, 2).contiguous().view(B, H * W, D)

        out = (h_fwd + h_bwd + v_fwd + v_bwd) / 4.0
        return out.view(B, H, W, D)


# ==========================================
# 4. ENCODER LAYER & TWO-STREAM ENCODER
# ==========================================
class VisualBranchEncoderLayer(nn.Module):
    def __init__(self, d_model, branch_type, d_state=16):
        super().__init__()
        self.norm_core = nn.LayerNorm(d_model)
        self.core = SpatialMambaBlock(d_model, d_state) if branch_type == 'spatial' else ContextMambaBlock(d_model, d_state)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, spatial_pe=None):
        # x: (B, H, W, D)
        residual = x
        x_norm = self.norm_core(x)
        if spatial_pe is not None:
            # spatial_pe: (1, H, W, D) — broadcast theo B
            x_norm = x_norm + spatial_pe

        x = residual + self.core(x_norm)
        return x + self.mlp(self.norm_mlp(x))


class TwoStreamVisualEncoder(nn.Module):
    def __init__(self, d_model, num_blocks=2, d_state=16):
        super().__init__()
        self.spatial_branch = nn.ModuleList([
            VisualBranchEncoderLayer(d_model, 'spatial', d_state) for _ in range(num_blocks)
        ])
        self.context_branch = nn.ModuleList([
            VisualBranchEncoderLayer(d_model, 'context', d_state) for _ in range(num_blocks)
        ])
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        self.norm_fusion = nn.LayerNorm(d_model)

    def forward(self, spatial_features, context_features, spatial_pe):
        # spatial_features, context_features: (B, H, W, D)
        x_sp, x_ctx = spatial_features, context_features
        for layer in self.spatial_branch:
            x_sp = layer(x_sp, spatial_pe)
        for layer in self.context_branch:
            x_ctx = layer(x_ctx, spatial_pe)

        x_fused = self.norm_fusion(
            self.fusion_proj(torch.cat([x_sp, x_ctx], dim=-1))
        )  # (B, H, W, D)
        return x_fused


# ==========================================
# 5. MÔ HÌNH CHÍNH: INTENT PREDICTOR (ảnh tĩnh)
# ==========================================
class IntentPredictor(nn.Module):
    def __init__(self, d_model=256, num_blocks=3):
        super().__init__()
        self.d_model = d_model
        # 0: go straight, 1: turn right, 2: turn left
        self.num_classes = 3

        self.video_backbone = CustomVideoBackbone(d_model=d_model)
        self.visual_encoder = TwoStreamVisualEncoder(d_model=d_model, num_blocks=num_blocks)

        self.intent_cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model, self.num_classes)
        )

    def forward(self, crop_img, full_img):
        """
        Args:
            crop_img: (B, C, H, W) — ảnh crop vùng người đi bộ
            full_img: (B, C, H, W) — ảnh toàn cảnh
        Returns:
            dict với logits và visual_features
        """
        # 1. Trích xuất đặc trưng từ ảnh tĩnh
        spatial_x, spatial_pe = self.video_backbone(crop_img)   # (B, H, W, D), (1, H, W, D)
        context_x, _          = self.video_backbone(full_img)   # (B, H, W, D)

        # 2. Mã hóa hai nhánh
        mamba_out = self.visual_encoder(spatial_x, context_x, spatial_pe)  # (B, H, W, D)

        # 3. Global average pooling không gian
        pooled_features = mamba_out.mean(dim=[1, 2])  # (B, D)

        # 4. Phân loại
        logits = self.intent_cls(pooled_features)  # (B, num_classes)

        return {
            "logits": logits,
            "visual_features": pooled_features,
        }


