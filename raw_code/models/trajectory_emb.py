import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

from models.backbone import CustomVideoBackbone


# ==========================================
# 2. CONTINUOUS-TIME DYNAMICS (CfC)
# ==========================================
class SOTALiquidCfCCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear_x = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(hidden_size, hidden_size)
        self.fc_candidate = nn.Linear(hidden_size, hidden_size)
        self.fc_tau = nn.Linear(hidden_size, hidden_size)
        self.ln_z = nn.LayerNorm(hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)

    def forward(self, x, h_prev, dt=1.0):
        z_raw = self.linear_x(x) + self.linear_h(h_prev)
        z = torch.tanh(self.ln_z(z_raw))
        candidate = torch.tanh(self.ln_c(self.fc_candidate(z)))
        tau = torch.clamp(F.softplus(self.fc_tau(z)), min=1e-4, max=100.0)
        gate = torch.exp(-dt / tau)
        return h_prev * gate + candidate * (1.0 - gate)

class DataDrivenCFCTemporalBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.cfc_cell = SOTALiquidCfCCell(input_size=d_model, hidden_size=d_model)
        self.dt_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1), nn.Softplus()
        )
        nn.init.constant_(self.dt_predictor[2].bias, -3.0)

    def forward(self, x):
        B, T, H, W, D = x.shape
        x_permuted = x.permute(0, 2, 3, 1, 4).contiguous() 
        x_temporal = x_permuted.view(B * H * W, T, D)
        
        h = torch.zeros(B * H * W, D, device=x.device, dtype=x.dtype)
        out_seq = []
        for t in range(T):
            x_t = x_temporal[:, t, :]
            dt = self.dt_predictor(x_t) + 1e-4 
            h = self.cfc_cell(x_t, h, dt)
            out_seq.append(h)
            
        out_cfc = torch.stack(out_seq, dim=1)
        out = out_cfc.view(B, H, W, T, D)
        return out.permute(0, 3, 1, 2, 4).contiguous().view(B, T, H, W, D)

# ==========================================
# 3. END-TO-END MODEL (VideoTrajectoryEmbedder)
# ==========================================
class VideoTrajectoryEmbedder(nn.Module):
    def __init__(self, d_model=256, text_embed_dim=512, num_classes=100, queue_size=1024):
        super().__init__()
        # 1. Khối Trích xuất Đặc trưng
        self.backbone = CustomVideoBackbone(d_model=d_model)
        self.temporal_cfc = DataDrivenCFCTemporalBlock(d_model=d_model)
        
        # 2. Khối Chiếu Vector
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, text_embed_dim)
        )
        nn.init.xavier_uniform_(self.proj_head[0].weight)
        nn.init.xavier_uniform_(self.proj_head[3].weight)
        
        # 3. Phân loại ID cho Re-ID Loss
        self.classifier = nn.Linear(text_embed_dim, num_classes)
        
        # 4. Hàng đợi Contrastive (XBM)
        self.queue_size = queue_size
        self.register_buffer("image_queue", torch.randn(text_embed_dim, queue_size))
        self.image_queue = F.normalize(self.image_queue, p=2, dim=0)
        
        self.register_buffer("text_queue", torch.randn(text_embed_dim, queue_size))
        self.text_queue = F.normalize(self.text_queue, p=2, dim=0)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, video_x):
        feat_3d, feature_to_visualize, spatial_pe, offsets = self.backbone(video_x)
        cfc_out = self.temporal_cfc(feat_3d)
        
        traj_feat = cfc_out.mean(dim=(2, 3)) # Global Average Pooling (Spatial)
        video_embed_raw = traj_feat.mean(dim=1) # Pooling Temporal
        
        video_embed = self.proj_head(video_embed_raw)
        video_embed_norm = F.normalize(video_embed, p=2, dim=-1)
        
        cls_logits = self.classifier(video_embed_norm)
        
        return video_embed_norm, cls_logits, offsets, feature_to_visualize

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        batch_size = image_feat.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0
        
        self.image_queue[:, ptr:ptr + batch_size] = image_feat.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feat.T
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size
