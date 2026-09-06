import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

# --- CORE MAMBA BLOCK ---
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

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
        y = selective_scan_fn(x_conv.transpose(1, 2), delta.transpose(1, 2), A, 
                              B_mat.transpose(1, 2), C_mat.transpose(1, 2), 
                              self.D_param.float(), delta_softplus=True).transpose(1, 2)
        return self.out_proj(y * F.silu(z_br))

# --- COMPONENTS ---
class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)

class AdvancedSpatialFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fusion_proj = nn.Linear(d_model * 4, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_fwd, h_bwd, v_fwd, v_bwd):
        combined = torch.cat([h_fwd, h_bwd, v_fwd, v_bwd], dim=-1)
        return self.norm(self.fusion_proj(combined))

class SpatialMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.core_h_fwd = MambaCore(d_model, d_state)
        self.core_h_bwd = MambaCore(d_model, d_state)
        self.core_v_fwd = MambaCore(d_model, d_state)
        self.core_v_bwd = MambaCore(d_model, d_state)
        self.fusion = AdvancedSpatialFusion(d_model)

    def forward(self, x_norm):
        B, T, H, W, D = x_norm.shape
        x_flat = x_norm.view(B * T, H * W, D)
        h_fwd = self.core_h_fwd(x_flat)
        h_bwd = self.core_h_bwd(x_flat.flip(dims=[1])).flip(dims=[1])
        
        x_v = x_norm.permute(0, 1, 3, 2, 4).contiguous().view(B * T, H * W, D)
        v_fwd = self.core_v_fwd(x_v).view(B * T, W, H, D).transpose(1, 2).reshape(B * T, H * W, D)
        v_bwd = self.core_v_bwd(x_v.flip(dims=[1])).flip(dims=[1]).view(B * T, W, H, D).transpose(1, 2).reshape(B * T, H * W, D)
        
        return self.fusion(h_fwd, h_bwd, v_fwd, v_bwd).view(B, T, H, W, D)

# --- BACKBONE & CLASSIFIER ---
class RobustColorDenseNet(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.color_transform = nn.Conv3d(3, 3, kernel_size=1)
        full_densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.early_extractor = nn.Sequential(*list(full_densenet.features.children())[:9])
        self.channel_attn = ECABlock(1024)
        self.patch_embed = nn.Sequential(nn.Conv2d(1024, d_model, 1), nn.GroupNorm(1, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.color_transform(x)
        B, C, T, H, W = x.shape
        feat = self.early_extractor(x.transpose(1, 2).reshape(B * T, C, H, W))
        feat = self.channel_attn(feat)
        out = self.patch_embed(feat).view(B, T, -1, feat.shape[-2], feat.shape[-1]).permute(0, 1, 3, 4, 2)
        return self.norm(out)

class CosineClassifier(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1)) * self.scale

# --- FULL MODEL ---
class PureColorMambaClassifier(nn.Module):
    def __init__(self, num_colors=11, d_model=256):
        super().__init__()
        self.backbone = RobustColorDenseNet(d_model)
        self.mamba_layer = SpatialMambaBlock(d_model)
        self.attn_pool = nn.Sequential(nn.Linear(d_model, d_model//2), nn.Tanh(), nn.Linear(d_model//2, 1))
        self.classifier = CosineClassifier(d_model, num_colors)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(2)
        x = self.backbone(x)
        x = self.mamba_layer(x)
        
        # Attention Pooling
        attn = F.softmax(self.attn_pool(x), dim=2)
        pooled = torch.sum(x * attn, dim=(1, 2, 3))
        return self.classifier(pooled)