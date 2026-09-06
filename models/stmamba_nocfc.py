import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 0. KHỞI TẠO MODULE MAMBA SSM
# ==========================================
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
    print(" Cảnh báo: mamba_ssm chưa được cài đặt. Không thể chạy module Mamba.")

# ==========================================
# 1. CORE MODULES (LÕI MAMBA CHUẨN)
# ==========================================
class MambaCore(nn.Module):
    """Lõi Mamba 1D chuẩn. Trọng số được chia sẻ (Weight Sharing) để quét nhiều hướng."""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        
        self.x_proj = nn.Linear(d_model, d_model + 2 * d_state)
        # Khởi tạo A_log theo chuẩn để tránh nổ gradient
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
            raise RuntimeError("mamba_ssm is required to run MambaCore.")

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
# 2. KHỐI KHÔNG GIAN & BỐI CẢNH
# ==========================================
class SpatialMambaBlock(nn.Module):
    """Quét 4 hướng thuần túy trên Visual Features (Không Cross-Attention Text)"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.spatial_conv2d = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.spatial_core = MambaCore(d_model, d_state) 
        
    def forward(self, x_norm):
        B, T, H, W, D = x_norm.shape
        
        # Biến đổi không gian để quét
        x_2d = x_norm.view(B * T, H, W, D).permute(0, 3, 1, 2).contiguous()
        x_conv = x_2d + self.spatial_conv2d(x_2d)
        x_base = x_conv.permute(0, 2, 3, 1).view(B * T, H * W, D)
        
        # Quét 4 hướng (Horizontal & Vertical)
        h_fwd = self.spatial_core(x_base)
        h_bwd = self.spatial_core(x_base.flip([1])).flip([1])
        
        x_v = x_base.view(B * T, H, W, D).transpose(1, 2).reshape(B * T, H * W, D)
        v_fwd = self.spatial_core(x_v)
        v_bwd = self.spatial_core(x_v.flip([1])).flip([1])
        
        # Đưa V-scan về lại không gian chuẩn
        v_fwd = v_fwd.view(B * T, W, H, D).transpose(1, 2).reshape(B * T, H * W, D)
        v_bwd = v_bwd.view(B * T, W, H, D).transpose(1, 2).reshape(B * T, H * W, D)
        
        out_spatial = (h_fwd + h_bwd + v_fwd + v_bwd) / 4.0
        return out_spatial.view(B, T, H, W, D)

class ContextMambaBlock(nn.Module):
    """Quét bối cảnh nền 1 chiều (Thuần Visual)"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.context_core = MambaCore(d_model, d_state)

    def forward(self, x_norm):
        B, T, H, W, D = x_norm.shape
        seq_ctx = x_norm.view(B, T * H * W, D)
        
        out_ctx_fwd = self.context_core(seq_ctx)
        out_ctx_bwd = self.context_core(seq_ctx.flip([1])).flip([1])
        
        x_final = out_ctx_fwd + out_ctx_bwd
        return x_final.view(B, T, H, W, D)

# ==========================================
# 3. KHỐI THỜI GIAN (CHỈ CÒN MAMBA CHO ABLATION)
# ==========================================
class TemporalMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.temporal_core = MambaCore(d_model, d_state)

    def forward(self, x):
        B, T, H, W, D = x.shape
        x_permuted = x.permute(0, 2, 3, 1, 4).contiguous() 
        x_temporal = x_permuted.view(B * H * W, T, D)
        
        out_t_fwd = self.temporal_core(x_temporal)
        out_t_bwd = self.temporal_core(x_temporal.flip([1])).flip([1])
        
        out = (out_t_fwd + out_t_bwd).view(B, H, W, T, D)
        return out.permute(0, 3, 1, 2, 4).contiguous().view(B, T, H, W, D)

class AblatedTemporalBlock(nn.Module):
    """Khối thời gian chỉ sử dụng Mamba (Đã loại bỏ CfC và Fusion Gate cho Ablation Study)"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.mamba_temporal = TemporalMambaBlock(d_model, d_state)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        out_mamba = self.mamba_temporal(x)
        return self.norm_out(out_mamba)

# ==========================================
# 4. VISUAL ENCODER TỔNG
# ==========================================
class VisualBranchEncoderLayer(nn.Module):
    def __init__(self, d_model, branch_type, d_state=16):
        super().__init__()
        self.branch_type = branch_type
        self.norm_core = nn.LayerNorm(d_model)
        
        if branch_type == 'spatial':
            self.core = SpatialMambaBlock(d_model, d_state)
        else:
            self.core = ContextMambaBlock(d_model, d_state)
            
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, spatial_pe):
        residual = x
        x_norm = self.norm_core(x)
        
        # Bổ sung Positional Encoding trước khi vào core
        if spatial_pe is not None:
            B, T, H, W, D = x_norm.shape
            x_norm = x_norm + spatial_pe.view(-1, H*W, D).view(1, 1, H, W, D)

        x_mamba = self.core(x_norm) 
        x = residual + x_mamba
        
        residual_mlp = x
        x = residual_mlp + self.mlp(self.norm_mlp(x))
        return x

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
        
        # Đã cập nhật thành khối Ablated (chỉ có Mamba)
        self.global_temporal = AblatedTemporalBlock(d_model, d_state)
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, spatial_features, context_features, spatial_pe):
        x_sp = spatial_features
        x_ctx = context_features

        for layer in self.spatial_branch:
            x_sp = layer(x_sp, spatial_pe)

        for layer in self.context_branch:
            x_ctx = layer(x_ctx, spatial_pe)

        merged = torch.cat([x_sp, x_ctx], dim=-1)
        x_fused = self.norm_fusion(self.fusion_proj(merged))

        # Khai phá động lực học thời gian (chỉ với Mamba)
        x_final = self.global_temporal(x_fused) 
        x_final = self.norm_f(x_final)
        
        return x_final