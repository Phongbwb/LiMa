import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.sensing.neuromorphicfronted import NeuromorphicFrontend
from models.spatial_extraction.dyspatialextr import DynamicSpatialExtraction
from models.backbone.stmamba import DualModulatedSTMambaBlock
from models.localization.microlocalization import MicroLocalization
from models.memory.astropool import MemoryAndPostProcessing
from models.detection.fpn import TrueFPN, YOLOHead # Đã đổi thành TrueFPN

class LiMaVLM(nn.Module):
    def __init__(self, in_channels=3, d_model=128, d_text=512, patch_size=4):
        super().__init__()
        self.d_model = d_model

        # 1. FRONTEND
        self.frontend = NeuromorphicFrontend(in_channels)

        # 2. SPATIAL BACKBONE
        self.spatial_extractor = DynamicSpatialExtraction(
            in_channels=in_channels,
            embed_dim=d_model,
            patch_size=patch_size
        )

        # 3. ST-MAMBA
        self.view_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        self.st_mamba = DualModulatedSTMambaBlock(
            d_model=d_model,
            d_state=16,
            d_text=d_text
        )

        # 4. MICRO LOCALIZATION
        self.micro_loc = MicroLocalization(embed_dim=d_model, num_parts=5)

        # 5. MEMORY
        self.memory = MemoryAndPostProcessing(
            embed_dim=d_model,
            d_text=d_text
        )

        # 6. HEADS (Global Tracking & Contrastive)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()  # Ép tọa độ [0, 1]
        )

        self.cls_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_text) # Khớp với Text Embedding
        )

        # 7. DETECTION (Dense)
        self.fpn = TrueFPN(d_model, out_channels=d_model) # Đồng bộ channel
        self.det_head = YOLOHead(in_channels=d_model, num_classes=1)

    def forward(self, video, text_emb, use_checkpointing=False, prev_states=None):
        """
        video: [B, C, T, H, W]
        text_emb: [B, d_text]
        use_checkpointing: Bật True khi training để cứu VRAM
        prev_states: (h_prev, G_prev) dùng cho việc train các video cực dài theo chunk
        """
        B, C, T, H, W = video.shape

        # ========================
        # 1. FRONTEND (Cập nhật nhận aux_loss)
        # ========================
        video, frontend_pred_loss = self.frontend(video)

        # ========================
        # 2. SPATIAL EXTRACTION 
        # ========================
        if use_checkpointing and self.training:
            def custom_spatial_forward(v):
                return self.spatial_extractor(v)
            
            # Khắc phục lỗi reentrant của checkpoint trong PyTorch mới
            tokens, mask, feature_map = checkpoint(
                custom_spatial_forward, 
                video, 
                use_reentrant=False
            )
        else:
            tokens, mask, feature_map = self.spatial_extractor(video)
            
        # tokens: [B, L, D] -> [B, T, N, D]
        L = tokens.shape[1]
        N = L // T
        tokens_4d = tokens.view(B, T, N, self.d_model)

        # ========================
        # 3. GLOBAL & SPATIAL TOKEN 
        # ========================
        avg_tokens = tokens_4d.mean(dim=2)           # [B, T, D]
        max_tokens = tokens_4d.max(dim=2)[0]         # [B, T, D]
        
        temporal_input = avg_tokens + max_tokens     
        global_tokens = avg_tokens                   

        # ========================
        # 4. ST-MAMBA
        # ========================
        view_scores = self.view_predictor(temporal_input)

        if use_checkpointing and self.training:
             temporal_out, gate_act = checkpoint(
                 self.st_mamba, temporal_input, text_emb, view_scores,
                 use_reentrant=False
             )
        else:
             temporal_out, gate_act = self.st_mamba(temporal_input, text_emb, view_scores)

        # ========================
        # 5. MICRO LOCALIZATION
        # ========================
        global_flat = global_tokens.reshape(B * T, self.d_model)
        fine_feat, coords, vis = self.micro_loc(feature_map, global_flat)
        fine_feat = fine_feat.view(B, T, self.d_model)
        
        # [NEW]: Tính Diversity Loss ngay trong forward (chỉ khi train)
        div_loss = self.micro_loc.get_diversity_loss(coords) if self.training else 0.0

        # ========================
        # 6. MEMORY (LNN - Cập nhật nhận/trả states)
        # ========================
        temporal_input_refined = temporal_out + fine_feat
        temporal_refined, next_states = self.memory(
            temporal_input_refined, 
            text_emb, 
            states=prev_states
        )

        # ========================
        # 7. FINAL REPRESENTATION
        # ========================
        last_global = temporal_refined[:, -1]
        last_fine = fine_feat[:, -1]

        fused = torch.cat([last_global, last_fine], dim=-1)
        
        cls_feat = self.cls_head(fused)
        bbox = self.bbox_head(last_global)

        # ========================
        # 8. DETECTION HEAD
        # ========================
        pyramid = self.fpn(feature_map)
        det_out = self.det_head(pyramid)

        # Đóng gói toàn bộ Output và Auxiliary Losses
        return {
            "bbox": bbox,
            "cls_feat": cls_feat,
            "det": det_out,
            "mask": mask,
            "feat_seq": temporal_refined,
            "gate": gate_act,
            "coords": coords,
            "visibility": vis,
            "next_states": next_states,           # Phục vụ chunking
            "aux_frontend_loss": frontend_pred_loss, # Phục vụ backward
            "aux_div_loss": div_loss              # Phục vụ backward
        }