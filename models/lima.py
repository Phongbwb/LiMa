import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Giả định các module này đã được bạn định nghĩa ở các file tương ứng
from models.backbone import CustomVideoBackbone
from models.stmamba import DecoupledVLMambaBlock
from models.astropool import MemoryAndPostProcessing
from models.microlocal import MicroLocalization

class LiMaVLM(nn.Module):
    def __init__(self, d_model=256, d_text=512, num_blocks=3):
        super().__init__()
        self.d_model = d_model
        
        # ==========================================
        # 1. FEATURE EXTRACTORS & ALIGNMENT
        # ==========================================    
        self.video_backbone = CustomVideoBackbone(d_model=d_model)
        
        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_model),
            nn.LayerNorm(d_model)
        )
        
        # ==========================================
        # 2. CORE: VL-MAMBA REASONING
        # ==========================================
        self.blocks = nn.ModuleList([
            DecoupledVLMambaBlock(d_model) for _ in range(num_blocks)
        ])
        
        # ==========================================
        # 3. MACRO-LOCALIZATION HEADS (CenterNet Style)
        # ==========================================
        self.hm_head = nn.Sequential(nn.Conv2d(d_model, d_model//2, 3, padding=1), nn.ReLU(), nn.Conv2d(d_model//2, 1, 1), nn.Sigmoid())
        self.sz_head = nn.Sequential(nn.Conv2d(d_model, d_model//2, 3, padding=1), nn.ReLU(), nn.Conv2d(d_model//2, 2, 1))
        self.off_head = nn.Sequential(nn.Conv2d(d_model, d_model//2, 3, padding=1), nn.ReLU(), nn.Conv2d(d_model//2, 2, 1))

        # ==========================================
        # 4. MICRO-LOCALIZATION & LIQUID MEMORY
        # ==========================================
        self.micro_loc = MicroLocalization(embed_dim=d_model, num_parts=5)
        self.liquid_memory = MemoryAndPostProcessing(embed_dim=d_model, d_text=d_text)

        # ==========================================
        # 5. RETRIEVAL HEAD (Contrastive Space)
        # ==========================================
        self.video_to_clip = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_text) # Chiếu ngược về không gian d_text của CLIP
        )
        # Sửa np.log thành math.log để tránh phụ thuộc thư viện ngoài
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07)) 

    def forward(self, video_frames, text_tokens, states=None):
        """
        video_frames: [B, C, T, H, W] (C=3)
        text_tokens: [B, N, d_text] (Đầu ra từ Text Encoder như CLIP/RoBERTa)
        states: Tuple (h, G) cho Liquid Memory khi quét video dài
        """
        B, C, T, H, W = video_frames.shape

        # --- BƯỚC 1: Trích xuất đặc trưng Video & Text ---
        # x shape: [B, T, Hp, Wp, d_model]
        x = self.video_backbone(video_frames)
        text_emb = self.text_proj(text_tokens)

        
        # Lấy Text tổng quát để dùng cho Memory sau này
        text_global_raw = text_tokens.mean(dim=1) # [B, d_text]

        # --- BƯỚC 2: Mamba dung hợp Đa phương thức ---
        for block in self.blocks:
            x = block(x, text_emb)
            
        # Nén batch và time để đưa vào mạng CNN 2D
        x_2d = rearrange(x, 'b t h w d -> (b t) d h w')

        # --- BƯỚC 3: Đầu ra CenterNet (Macro Tracking) ---
        # Phục vụ cho Loss tính toán Bounding Box
        hm = self.hm_head(x_2d)
        sz = self.sz_head(x_2d)
        off = self.off_head(x_2d)

        # --- BƯỚC 4: Micro-Localization (Tìm chi tiết nhỏ) ---
        # Tạo global_tokens từ x_2d làm query tìm part
        global_tokens = x_2d.mean(dim=[2, 3]) # [B*T, d_model]
        
        # micro_feat: [B*T, d_model] - Đặc trưng xe cực kỳ sắc nét
        micro_feat, coords, vis = self.micro_loc(x_2d, global_tokens) 

        # --- BƯỚC 5: Ổn định qua Liquid Memory ---
        # Tái tạo lại trục thời gian T
        micro_seq = rearrange(micro_feat, '(b t) d -> b t d', b=B, t=T)
        
        # Lọc bỏ nhiễu do che khuất
        stable_seq, new_states = self.liquid_memory(micro_seq, text_global_raw, states=states) # [B, T, d_model]

        # --- BƯỚC 6: Đầu ra Video-Text Retrieval ---
        # Gom toàn bộ thời gian thành 1 vector duy nhất đại diện cho xe
        video_global = stable_seq.mean(dim=1) # [B, d_model]
        video_retrieval_feat = self.video_to_clip(video_global) # [B, d_text]

        # Chuẩn hóa để tính Contrastive Loss (InfoNCE)
        video_retrieval_feat = F.normalize(video_retrieval_feat, dim=-1)

        return {
            "tracking_heads": (hm, sz, off),
            "retrieval_feat": video_retrieval_feat,
            "micro_parts": (coords, vis),
            "memory_states": new_states
        }