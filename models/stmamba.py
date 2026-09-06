import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. KHỞI TẠO MODULE MAMBA & CFC (CÓ FALLBACK)
# ==========================================
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
    print("Cảnh báo: mamba_ssm chưa được cài đặt. Không thể chạy module Mamba chuẩn.")

try:
    from ncps.torch import CfC
except ImportError:
    CfC = None
    print("Cảnh báo: ncps chưa được cài đặt. Nhánh Trajectory sẽ dùng GRU thay thế.")

# ==========================================
# 1. CORE MODULES (LÕI MAMBA CHUẨN)
# ==========================================
class MambaCore(nn.Module):
    """Lõi Mamba 1D chuẩn."""
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
# 2. KHỐI VISUAL (SPATIAL & CONTEXT)
# ==========================================
class SpatialMambaBlock(nn.Module):
    """Quét 4 hướng chuẩn VideoMamba (4 lõi Mamba độc lập)"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.spatial_conv2d = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.core_h_fwd = MambaCore(d_model, d_state) 
        self.core_h_bwd = MambaCore(d_model, d_state) 
        self.core_v_fwd = MambaCore(d_model, d_state) 
        self.core_v_bwd = MambaCore(d_model, d_state) 
        
    def forward(self, x_norm):
        B, T, H, W, D = x_norm.shape
        
        x_2d = x_norm.view(B * T, H, W, D).permute(0, 3, 1, 2).contiguous()
        x_conv = x_2d + self.spatial_conv2d(x_2d) 
        
        # 1. Quét ngang
        x_h = x_conv.permute(0, 2, 3, 1).contiguous().view(B * T, H * W, D)
        h_fwd = self.core_h_fwd(x_h)
        h_bwd = self.core_h_bwd(x_h.flip(dims=[1])).flip(dims=[1])
        
        # 2. Quét dọc
        x_v = x_conv.permute(0, 3, 2, 1).contiguous().view(B * T, W * H, D)
        v_fwd_out = self.core_v_fwd(x_v)
        v_bwd_out = self.core_v_bwd(x_v.flip(dims=[1])).flip(dims=[1])
        
        v_fwd = v_fwd_out.view(B * T, W, H, D).transpose(1, 2).contiguous().view(B * T, H * W, D)
        v_bwd = v_bwd_out.view(B * T, W, H, D).transpose(1, 2).contiguous().view(B * T, H * W, D)
        
        # 3. Hợp nhất
        out_spatial = (h_fwd + h_bwd + v_fwd + v_bwd) / 4.0
        return out_spatial.view(B, T, H, W, D)

class ContextMambaBlock(nn.Module):
    """Quét bối cảnh nền Bi-directional (2 lõi Mamba độc lập)"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.core_fwd = MambaCore(d_model, d_state)
        self.core_bwd = MambaCore(d_model, d_state)

    def forward(self, x_norm):
        B, T, H, W, D = x_norm.shape
        seq_ctx = x_norm.view(B, T * H * W, D)
        out_ctx_fwd = self.core_fwd(seq_ctx)
        out_ctx_bwd = self.core_bwd(seq_ctx.flip(dims=[1])).flip(dims=[1])
        return ((out_ctx_fwd + out_ctx_bwd) / 2.0).view(B, T, H, W, D)

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

class VisualBranchEncoderLayer(nn.Module):
    def __init__(self, d_model, branch_type, d_state=16):
        super().__init__()
        self.norm_core = nn.LayerNorm(d_model)
        self.core = SpatialMambaBlock(d_model, d_state) if branch_type == 'spatial' else ContextMambaBlock(d_model, d_state)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))

    def forward(self, x, spatial_pe):
        residual = x
        x_norm = self.norm_core(x)
        if spatial_pe is not None:
            B, T, H, W, D = x_norm.shape
            x_norm = x_norm + spatial_pe.view(-1, H*W, D).view(1, 1, H, W, D)
        
        x = residual + self.core(x_norm) 
        return x + self.mlp(self.norm_mlp(x))

class TwoStreamVisualEncoder(nn.Module):
    def __init__(self, d_model, num_blocks=2, d_state=16):
        super().__init__()
        self.spatial_branch = nn.ModuleList([VisualBranchEncoderLayer(d_model, 'spatial', d_state) for _ in range(num_blocks)])
        self.context_branch = nn.ModuleList([VisualBranchEncoderLayer(d_model, 'context', d_state) for _ in range(num_blocks)])
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        self.norm_fusion = nn.LayerNorm(d_model)
        self.global_temporal = TemporalMambaBlock(d_model, d_state)
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, spatial_features, context_features, spatial_pe):
        x_sp, x_ctx = spatial_features, context_features
        for layer in self.spatial_branch: x_sp = layer(x_sp, spatial_pe)
        for layer in self.context_branch: x_ctx = layer(x_ctx, spatial_pe)
        
        # LƯU LẠI RAW CONTEXT EMBEDDING TRƯỚC KHI FUSION
        context_emb_raw = x_ctx 
        
        # Tiếp tục nhánh dung hợp Visual
        x_fused = self.norm_fusion(self.fusion_proj(torch.cat([x_sp, x_ctx], dim=-1)))
        visual_out = self.norm_f(self.global_temporal(x_fused)) 
        
        # Trả về cả output của Visual Branch và Context Emb (chưa dung hợp)
        return visual_out, context_emb_raw

# ==========================================
# 3. KHỐI TRAJECTORY (BBOX FEATURES)
# ==========================================
class FallbackCfC(nn.Module):
    """Module dự phòng nếu chưa cài đặt thư viện NCPS"""
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
    def forward(self, x):
        out, _ = self.rnn(x)
        return out

class StackedCfC(nn.Module):
    """
    Tự động xếp chồng nhiều lớp CfC (Do thư viện ncps không có tham số num_layers).
    """
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Lớp đầu tiên nhận input_size
        self.layers.append(CfC(input_size=input_size, units=hidden_size, batch_first=True))
        
        # Các lớp tiếp theo nhận hidden_size làm input
        for _ in range(num_layers - 1):
            self.layers.append(CfC(input_size=hidden_size, units=hidden_size, proj_size=hidden_size, batch_first=True))
            
    def forward(self, x):
        for layer in self.layers:
            # CfC trả về tuple (output, hidden_state), ta chỉ lấy output để truyền sang lớp tiếp theo
            x, _ = layer(x)
        return x

class BiMambaSequence(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.mamba_fwd = MambaCore(d_model, d_state)
        self.mamba_bwd = MambaCore(d_model, d_state)
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        fwd = self.mamba_fwd(x)
        bwd = self.mamba_bwd(x.flip([1])).flip([1])
        return self.norm(self.out_proj(torch.cat([fwd, bwd], dim=-1)))

class TrajectoryBranch(nn.Module):
    def __init__(self, input_dim=18, d_model=128, d_state=16):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim) 
        
        # ĐÃ SỬA: Gọi StackedCfC thay vì truyền trực tiếp num_layers vào CfC
        if CfC is not None:
            self.cfc = StackedCfC(input_size=input_dim, hidden_size=d_model, num_layers=2)
        else:
            self.cfc = FallbackCfC(input_size=input_dim, hidden_size=d_model, num_layers=2)
            
        self.norm_cfc = nn.LayerNorm(d_model)
        self.bi_mamba = BiMambaSequence(d_model, d_state)

    def forward(self, bbox_features):
        x = self.input_norm(bbox_features)
        x = self.norm_cfc(self.cfc(x))
        x = self.bi_mamba(x)
        return x.mean(dim=1) # -> [B, D]

