import torch
import torch.nn as nn
import torch.nn.functional as F
# Yêu cầu cài đặt: pip install mamba-ssm einops
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
    print("⚠️ Cảnh báo: Không tìm thấy mamba_ssm. SelectiveScan sẽ không hoạt động trên GPU.")
    
# ==========================================
# 2. CORE: BI-DIRECTIONAL HIERARCHICAL MAMBA
# ==========================================
class MambaCore(nn.Module):
    """Lõi Mamba thuần túy tái sử dụng được"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        self.x_proj = nn.Linear(d_model, d_model + 2 * d_state)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(d_model, 1)))
        self.D_param = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_br, z_br = xz.chunk(2, dim=-1)
        x_conv = F.silu(self.conv1d(x_br.transpose(1, 2))[:, :, :L]).transpose(1, 2)

        proj = self.x_proj(x_conv)
        delta, B_mat, C_mat = torch.split(proj, [D, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())

        y = selective_scan_fn(
            x_conv.transpose(1, 2).contiguous(), delta.transpose(1, 2).contiguous(), 
            A, B_mat.transpose(1, 2).contiguous(), 
            C_mat.transpose(1, 2).contiguous(), self.D_param.float(),
            delta_softplus=True
        ).transpose(1, 2)

        return self.out_proj(y * F.silu(z_br))

class BottleneckTextDecoupler(nn.Module):
    """
    Decoupler điểm ngọt: Vừa có tính phi tuyến để tách ngữ nghĩa tốt, 
    vừa là dạng bottleneck để hội tụ nhanh và không phá hỏng gradient.
    """
    def __init__(self, d_model, reduction=4):
        super().__init__()
        
        # d_hidden nhỏ (ví dụ 256 -> 64) ép mạng phải chắt lọc thông tin
        d_hidden = d_model // reduction
        
        # Nhánh Appearance: Tìm kiếm đặc điểm tĩnh
        self.app_proj = nn.Sequential(
            nn.Linear(d_model, d_hidden, bias=False),
            nn.SiLU(), # Hàm phi tuyến mượt mà, tốt hơn ReLU cho VLM
            nn.Linear(d_hidden, d_model, bias=False)
        )
        
        # Nhánh Motion: Tìm kiếm quỹ đạo
        self.motion_proj = nn.Sequential(
            nn.Linear(d_model, d_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(d_hidden, d_model, bias=False)
        )

        # Mẹo khởi tạo (Zero-Init): 
        # Khởi tạo lớp Linear thứ 2 bằng 0. Ban đầu nhánh này không tác động gì,
        # giúp Mamba ổn định ở các epoch đầu (warm-up), sau đó mới học dần cách tách text.
        nn.init.zeros_(self.app_proj[2].weight)
        nn.init.zeros_(self.motion_proj[2].weight)

    def forward(self, text_emb):
        text_global = text_emb.mean(dim=1) # [B, d_model]
        
        # Tách ngữ nghĩa qua Bottleneck
        text_app = self.app_proj(text_global)
        text_motion = self.motion_proj(text_global)
        
        # Residual connection: Cộng lại với vector gốc để không làm mất thông tin tổng thể
        return text_global + text_app, text_global + text_motion

class DecoupledVLMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.text_decoupler = BottleneckTextDecoupler(d_model)

        # 1. Nhánh Không gian (Appearance Tracking)
        self.norm_spatial = nn.LayerNorm(d_model)
        self.spatial_mamba = MambaCore(d_model, d_state)

        # 2. Nhánh Thời gian (Motion Tracking)
        self.norm_temporal = nn.LayerNorm(d_model)
        self.temporal_forward = MambaCore(d_model, d_state)
        self.temporal_backward = MambaCore(d_model, d_state)

    def forward(self, x, text_tokens):
        # x: [B, T, H, W, D] (Output từ CustomVideoBackbone)
        # text_tokens: [B, N, D]
        B, T, H, W, D = x.shape

        # --- 0. TÁCH NGỮ CẢNH VĂN BẢN ---
        text_app, text_motion = self.text_decoupler(text_tokens)
        # --- 1. SPATIAL SCAN (Tìm "Ngoại hình" chiếc xe) ---
        x_spatial = x.contiguous().view(B * T, H * W, D)
        
        # Lặp text_app cho T frame: [B, D] -> [B, 1, D] -> [B, T, 1, D] -> [B*T, 1, D]
        text_app_prefix = text_app.unsqueeze(1).unsqueeze(2).expand(B, T, 1, D).reshape(B * T, 1, D)
        
        # Nối Text App làm token đầu tiên: [B*T, 1 + H*W, D]
        seq_spatial = torch.cat([text_app_prefix, x_spatial], dim=1)
        
        # Đưa qua Mamba và bỏ token Text đầu tiên đi
        out_spatial = self.spatial_mamba(self.norm_spatial(seq_spatial))[:, 1:, :] 
        
        # Cộng Residual
        x = x + out_spatial.view(B, T, H, W, D)

        # --- 2. TEMPORAL SCAN (Theo dõi "Chuyển động" chiếc xe) ---
        x_temporal = x.permute(0, 2, 3, 1, 4).contiguous().view(B * H * W, T, D)
        
        # Lặp text_motion cho H*W pixel: [B, D] -> [B, 1, D] -> [B, H*W, 1, D] -> [B*H*W, 1, D]
        text_motion_prefix = text_motion.unsqueeze(1).unsqueeze(2).expand(B, H * W, 1, D).reshape(B * H * W, 1, D)

        # Hướng Tiến (Forward)
        seq_forward = torch.cat([text_motion_prefix, x_temporal], dim=1) 
        out_forward = self.temporal_forward(self.norm_temporal(seq_forward))[:, 1:, :] # Bỏ token Text

        # Hướng Lùi (Backward)
        x_temporal_rev = torch.flip(x_temporal, dims=[1])
        seq_backward = torch.cat([text_motion_prefix, x_temporal_rev], dim=1)
        out_backward = self.temporal_backward(self.norm_temporal(seq_backward))[:, 1:, :]
        out_backward = torch.flip(out_backward, dims=[1]) # Lật xuôi lại

        # Trộn đặc trưng
        x_temporal_out = x_temporal + out_forward + out_backward

        # Trả về shape gốc [B, T, H, W, D]
        x_out = x_temporal_out.view(B, H, W, T, D).permute(0, 3, 1, 2, 4)
        return x_out