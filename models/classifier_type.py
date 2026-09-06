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
# 0. KHỞI TẠO MODULE MAMBA (FALLBACK)
# ==========================================
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
    print("Cảnh báo: mamba_ssm chưa được cài đặt. Sẽ dùng Fallback.")


# ==========================================
# 1. CBAM: CHANNEL + SPATIAL ATTENTION
#    Giúp model tập trung vào vùng đặc trưng của xe
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ==========================================
# 2. MULTI-SCALE FEATURE PYRAMID
#    Backbone: EfficientNet-B0 (~5.3M params, nhẹ hơn ResNet50 ~5x)
#    Trích đặc trưng từ 3 stage: s3/s4/s5 của EfficientNet
#
#    EfficientNet-B0 channel map (features[i]):
#      features[0] : stem         —  32ch, stride 2
#      features[1] : MBConv block — stage 1
#      features[2] : MBConv block — stage 2,  40ch, stride 8   <- P3
#      features[3] : MBConv block — stage 3,  80ch, stride 16  <- P4
#      features[4..8]: stages còn lại
#      features[8] : stage cuối  — 320ch, stride 32            <- P5
# ==========================================
class MultiScaleFeaturePyramid(nn.Module):
    # Channels xuất ra chuẩn xác từ torchvision.models.efficientnet_b0
    _EFFNET_CHANNELS = {
        "p3": 40,    # features[3]
        "p4": 80,    # features[4]
        "p5": 320,   # features[7]
    }

    def __init__(self, d_model=256):
        super().__init__()

        # Sử dụng API load weights mới của PyTorch để tránh lỗi tải lại liên tục
        effnet = models.efficientnet_b0(weights=None)  # khởi tạo không load weights
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth',
            map_location='cpu',
            check_hash=False   # bỏ qua hash check
        )
        effnet.load_state_dict(state_dict)

        # CẮT (SLICE) LẠI INDEX CHO ĐÚNG:
        # P3: lấy từ đầu đến hết features[3] -> Output: 40 ch, H/8
        self.stage_p3 = effnet.features[:4]  
        
        # P4: lấy tiếp features[4] -> Output: 80 ch, H/16
        self.stage_p4 = effnet.features[4:5] 
        
        # P5: lấy tiếp từ features[5] đến hết features[7] -> Output: 320 ch, H/32
        # Lưu ý: Không lấy features[8] vì nó ra tới 1280 channels
        self.stage_p5 = effnet.features[5:8] 

        # Freeze P3
        for p in self.stage_p3.parameters():
            p.requires_grad = False

        ch = self._EFFNET_CHANNELS

        # Lateral projections → d_model
        def _lateral(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, d_model, 1, bias=False),
                nn.GroupNorm(8, d_model),
                nn.ReLU()
            )

        self.lateral_p3 = _lateral(ch["p3"])
        self.lateral_p4 = _lateral(ch["p4"])
        self.lateral_p5 = _lateral(ch["p5"])

        # CBAM attention cho từng scale
        self.cbam_p3 = CBAM(d_model)
        self.cbam_p4 = CBAM(d_model)
        self.cbam_p5 = CBAM(d_model)

        # Fusion: cat 3 scale → d_model
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(d_model * 3, d_model, 1, bias=False),
            nn.GroupNorm(8, d_model),
            nn.ReLU()
        )

    def forward(self, x):
        # Tính toán đặc trưng theo chuỗi
        f3 = self.stage_p3(x)                   # (B, 40, H/8, W/8)
        f4 = self.stage_p4(f3)                  # (B, 80, H/16, W/16)
        f5 = self.stage_p5(f4)                  # (B, 320, H/32, W/32)

        # Lateral + CBAM attention
        p3 = self.cbam_p3(self.lateral_p3(f3))  # (B, 256, H/8, W/8)
        p4 = self.cbam_p4(self.lateral_p4(f4))  # (B, 256, H/16, W/16)
        p5 = self.cbam_p5(self.lateral_p5(f5))  # (B, 256, H/32, W/32)

        # Top-down upsample về kích thước P3
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5, size=p3.shape[-2:], mode='bilinear', align_corners=False)

        # Fuse 3 scale
        fused = self.fusion_proj(torch.cat([p3, p4_up, p5_up], dim=1))  # (B, 256, H/8, W/8)
        return fused

# ==========================================
# 3. VEHICLE DISCRIMINATIVE BACKBONE
#    Kết hợp EfficientNet-B0 FPN đa tỉ lệ + Fourier PE
# ==========================================
class VehicleDiscriminativeBackbone(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.fpn = MultiScaleFeaturePyramid(d_model)

        # Fourier positional encoding
        self.spatial_mlp = nn.Conv2d(18, d_model, 1)   # 18 = 2 + 2*4*2 features
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(d_model)

    def get_fourier_features(self, coords, num_bands=4):
        features = [coords]
        for freq in [2 ** i for i in range(num_bands)]:
            features.append(torch.sin(math.pi * freq * coords))
            features.append(torch.cos(math.pi * freq * coords))
        return torch.cat(features, dim=-1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out:        (B, Hp, Wp, D)
            spatial_pe: (1, Hp, Wp, D)
        """
        feat_2d = self.fpn(x)               # (B, D, Hp, Wp)
        _, D, Hp, Wp = feat_2d.shape

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, Hp, device=x.device),
            torch.linspace(-1, 1, Wp, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)                       # (1, Hp, Wp, 2)
        grid_fourier = self.get_fourier_features(grid).permute(0, 3, 1, 2)      # (1, 18, Hp, Wp)
        spatial_pe = self.spatial_mlp(grid_fourier).permute(0, 2, 3, 1)         # (1, Hp, Wp, D)

        out = feat_2d.permute(0, 2, 3, 1) + spatial_pe * self.pe_scale          # (B, Hp, Wp, D)
        return self.norm(out), spatial_pe


# ==========================================
# 4. LÕI MAMBA 1D
# ==========================================
class MambaCore(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.x_proj = nn.Linear(d_model, d_model + 2 * d_state)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float().repeat(d_model, 1))
        )
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
# 5. SPATIAL MAMBA BLOCK (quét 4 hướng)
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

        x_2d = x_norm.permute(0, 3, 1, 2).contiguous()        # (B, D, H, W)
        x_conv = x_2d + self.spatial_conv2d(x_2d)              # (B, D, H, W)

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
# 6. SPATIAL ENCODER LAYER
# ==========================================
class SpatialEncoderLayer(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm_core = nn.LayerNorm(d_model)
        self.core = SpatialMambaBlock(d_model, d_state)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, spatial_pe=None):
        # x: (B, H, W, D)
        x_norm = self.norm_core(x)
        if spatial_pe is not None:
            x_norm = x_norm + spatial_pe          # broadcast theo B
        x = x + self.core(x_norm)
        x = x + self.mlp(self.norm_mlp(x))
        return x


# ==========================================
# 7. SPATIAL MAMBA ENCODER (1 nhánh, không context)
# ==========================================
class SpatialMambaEncoder(nn.Module):
    def __init__(self, d_model, num_blocks=3, d_state=16):
        super().__init__()
        self.layers = nn.ModuleList([
            SpatialEncoderLayer(d_model, d_state) for _ in range(num_blocks)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x, spatial_pe):
        """
        Args:
            x:          (B, H, W, D)
            spatial_pe: (1, H, W, D)
        Returns:
            (B, H, W, D)
        """
        for layer in self.layers:
            x = layer(x, spatial_pe)
        return self.norm_out(x)


# ==========================================
# 8. GEM POOLING
#    Nhạy hơn average pooling với đặc trưng cục bộ
#    (điểm nổi bật trên xe: biển số, logo, đèn)
# ==========================================
class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x):
        # x: (B, H, W, D) — pool theo H, W
        x = x.permute(0, 3, 1, 2)  # (B, D, H, W)
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            output_size=(1, 1)
        ).pow(1.0 / self.p).squeeze(-1).squeeze(-1)  # (B, D)


# ==========================================
# 9. MÔ HÌNH CHÍNH: INTENT PREDICTOR
#    Chỉ dùng crop_img (nhánh spatial duy nhất)
# ==========================================
class IntentPredictor(nn.Module):
    def __init__(self, d_model=256, num_blocks=3, num_classes=3):
        """
        Args:
            d_model:     chiều đặc trưng
            num_blocks:  số SpatialEncoderLayer
            num_classes: số lớp phân loại
                         (ví dụ: 0=đi thẳng, 1=rẽ phải, 2=rẽ trái)
        """
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Backbone chuyên biệt phân biệt xe
        self.backbone = VehicleDiscriminativeBackbone(d_model=d_model)

        # Encoder Mamba chỉ 1 nhánh spatial
        self.spatial_encoder = SpatialMambaEncoder(
            d_model=d_model, num_blocks=num_blocks
        )

        # GeM pooling thay cho average pooling
        self.gem_pool = GeMPooling(p=3.0)

        # Head phân loại với feature normalization
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes)
        )

    def extract_features(self, img):
        """
        Trích xuất đặc trưng đã được encode.
        Dùng được cho metric learning / ArcFace.
        Returns: (B, D) L2-normalized features
        """
        feat, spatial_pe = self.backbone(img)           # (B, H, W, D), (1, H, W, D)
        encoded = self.spatial_encoder(feat, spatial_pe) # (B, H, W, D)
        pooled = self.gem_pool(encoded)                  # (B, D)
        return F.normalize(pooled, dim=-1)               # L2 normalize

    def forward(self, crop_img):
        """
        Args:
            crop_img: (B, C, H, W) — ảnh crop vùng xe / người đi bộ
        Returns:
            dict với logits, features (normalized), raw_features
        """
        feat, spatial_pe = self.backbone(crop_img)           # (B, H, W, D)
        encoded = self.spatial_encoder(feat, spatial_pe)      # (B, H, W, D)
        pooled = self.gem_pool(encoded)                       # (B, D)

        norm_features = F.normalize(pooled, dim=-1)           # (B, D) dùng cho metric
        logits = self.classifier(pooled)                      # (B, num_classes)

        return {
            "logits": logits,
            "features": norm_features,      # L2-normalized, dùng cho ArcFace / triplet
            "raw_features": pooled,         # trước normalize
        }

