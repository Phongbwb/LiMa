import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# ==========================================
# 1. LÕI SSM VỚI COSINE SIMILARITY GATE
# ==========================================
class CosineModulatedSelectiveScan(nn.Module):
    def __init__(self, d_model, d_state=16, d_text=512):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2) # Tách nhánh x và z
        
        # Causal Conv1d
        self.conv1d = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, 
            kernel_size=3, padding=0, groups=d_model
        )
        self.x_proj = nn.Linear(d_model, d_model + 2 * d_state)

        # Khởi tạo ma trận A tĩnh và D
        A_init = torch.arange(1, d_state + 1).float().repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D_param = nn.Parameter(torch.ones(d_model))

        # --- COSINE ALIGNMENT GATING ---
        # "Ống phóng to": Nâng đặc trưng Mamba lên bằng với CLIP (512D)
        self.video_to_clip_space = nn.Linear(d_model, d_text)
        
        # Tham số scale nhiệt độ tự học (Giống trong paper gốc của CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # HACK KHỞI TẠO: Thêm một bias tĩnh = +2.0 vào phương trình Cổng.
        # Lý do: Lúc đầu Cosine ~ 0, nhân với logit_scale sẽ ra 0. Sigmoid(0) = 0.5 (Kẹt cổng).
        # Bias +2.0 ép Sigmoid nhích lên ~0.88, "mở toang cửa" ở Epoch 0 để Backbone dễ thở.
        self.gate_bias = nn.Parameter(torch.tensor(2.0))

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, text_emb, temperature=1.0):
        B_dim, L, D = x.shape
        residual = x

        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # Causal Conv1D
        x_branch = x_branch.transpose(1, 2)
        x_pad = F.pad(x_branch, (2, 0))
        x_conv = self.conv1d(x_pad)[:, :, :L]
        x_conv = F.silu(x_conv).transpose(1, 2)

        # ==========================================
        # ĐÂY LÀ TRÁI TIM CỦA COSINE GATE
        # ==========================================
        # 1. Chiếu Video [B, L, 256] -> Không gian CLIP [B, L, 512]
        video_proj = self.video_to_clip_space(x_conv) 
        
        # 2. Chuẩn hóa Vector (L2 Norm) để đo góc
        video_norm = F.normalize(video_proj, p=2, dim=-1)
        text_norm = F.normalize(text_emb, p=2, dim=-1).unsqueeze(1) # [B, 1, 512]
        
        # 3. Tính Cosine Similarity (Tích vô hướng). Kết quả: [B, L, 1]
        sim_scores = torch.sum(video_norm * text_norm, dim=-1, keepdim=True)
        
        # 4. Tính Cổng: Áp dụng Scale tự động + Bias "Mở toang" + Temperature Annealing
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        gate_logits = (sim_scores * logit_scale + self.gate_bias) / temperature
        gate = torch.sigmoid(gate_logits) # Output: [B, L, 1] (Một hệ số duy nhất cho mỗi token)
        # ==========================================

        # Chiếu ra tham số Mamba (B, C, Delta)
        proj = self.x_proj(x_conv)
        delta_base, B_base, C_base = torch.split(proj, [D, self.d_state, self.d_state], dim=-1)

        # Áp dụng cổng vô hướng [B, L, 1] đóng/mở TOÀN BỘ d_state chiều của B và C
        B_mod = B_base * gate
        C_mod = C_base * gate
        delta_mod = delta_base # Không cần Gate Delta, để SSM giữ nhịp độ nội tại

        # Chuẩn bị Tensor cho CUDA Kernel
        A_mat = -torch.exp(self.A_log.float()).contiguous()
        u = x_conv.transpose(1, 2).contiguous().float()
        delta_mod = delta_mod.transpose(1, 2).contiguous().float()
        B_mod = B_mod.transpose(1, 2).contiguous().float()
        C_mod = C_mod.transpose(1, 2).contiguous().float()
        D_vec = self.D_param.float().contiguous()

        # Quét siêu tốc
        y = selective_scan_fn(
            u, delta_mod, A_mat, B_mod, C_mod, D_vec,
            z=None, delta_bias=None, delta_softplus=True
        )
        y = y.transpose(1, 2)

        # Nhánh Gating Z của kiến trúc Mamba gốc
        y = y * F.silu(z_branch)
        y = self.out_proj(y) + residual

        return y, gate

# ==========================================
# 2. KHỐI ĐỊNH TUYẾN KHÔNG GIAN - THỜI GIAN
# ==========================================
class STMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_text=512):
        super().__init__()
        self.spatial_mamba = CosineModulatedSelectiveScan(d_model, d_state, d_text)
        self.temporal_mamba = CosineModulatedSelectiveScan(d_model, d_state, d_text)

    def forward(self, video_grid, text_emb, temperature=1.0):
        B, T, H, W, D = video_grid.shape
        
        # --- S-Mamba (Không gian) ---
        x_spatial = rearrange(video_grid, 'b t h w d -> (b t) (h w) d')
        text_spatial = text_emb.repeat_interleave(T, dim=0)
        
        out_spatial, gate_S = self.spatial_mamba(x_spatial, text_spatial, temperature)
        out_spatial = rearrange(out_spatial, '(b t) (h w) d -> b t h w d', b=B, t=T, h=H, w=W)
        
        # --- T-Mamba (Thời gian) ---
        x_temporal = rearrange(video_grid, 'b t h w d -> (b h w) t d')
        text_temporal = text_emb.repeat_interleave(H*W, dim=0)
        
        out_temporal, gate_T = self.temporal_mamba(x_temporal, text_temporal, temperature)
        out_temporal = rearrange(out_temporal, '(b h w) t d -> b t h w d', b=B, h=H, w=W)
        
        # Gộp thông tin
        out = video_grid + out_spatial + out_temporal
        
        # Gom các Cổng vô hướng lại
        return out, [gate_S, gate_T]

# ==========================================
# 3. LỚP CENTER-POINT HEAD
# ==========================================
class CenterPointHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, 1, kernel_size=1),
            nn.Sigmoid() 
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, 2, kernel_size=1)
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, 2, kernel_size=1)
        )

    def forward(self, x):
        return self.heatmap_head(x), self.size_head(x), self.offset_head(x)

# ==========================================
# 4. KIẾN TRÚC TỔNG THỂ LI-MA VLM
# ==========================================
class LiMaVLM(nn.Module):
    def __init__(self, d_model=256, d_state=16, d_text=512, num_blocks=4):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 1, 16, 16, d_model))
        
        self.blocks = nn.ModuleList([
            STMambaBlock(d_model, d_state, d_text) for _ in range(num_blocks)
        ])
        
        self.head = CenterPointHead(d_model)

    def forward(self, video_features, text_features, temperature=1.0):
        x = video_features + self.pos_embed
        
        collected_gates = []
        for block in self.blocks:
            x, gates = block(x, text_features, temperature)
            collected_gates.extend(gates)
            
        B, T, H, W, D = x.shape
        x_head = rearrange(x, 'b t h w d -> (b t) d h w')
        
        heatmap, size, offset = self.head(x_head)
        return heatmap, size, offset, collected_gates

# ==========================================
# 5. HỆ THỐNG HÀM LOSS (Khớp với Cosine)
# ==========================================
def focal_loss(pred, target, alpha=2, beta=4):
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, beta)
    
    pos_loss = torch.log(pred + 1e-6) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred + 1e-6) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        return -neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos

def gate_sparsity_loss(gates_list):
    """ Ép cổng Cosine dạt về 0 hoặc 1 """
    loss = 0
    for gate in gates_list:
        loss += torch.mean(gate * (1.0 - gate))
    return loss / len(gates_list)

