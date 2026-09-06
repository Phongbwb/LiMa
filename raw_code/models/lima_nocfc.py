import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from models.backbone import CustomVideoBackbone
from models.stmamba import TwoStreamVisualEncoder 
# Không import TrajectoryBranch nữa

class LiMaVLM(nn.Module):
    def __init__(self, d_model=256, d_text_in=512, num_blocks=3, num_classes=2155, queue_size=1024):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.queue_size = queue_size

        # ==========================================
        # 1. VISUAL & TEXT ENCODERS (KHÔNG CÓ TRAJECTORY)
        # ==========================================
        self.video_backbone = CustomVideoBackbone(d_model=d_model)
        self.visual_encoder = TwoStreamVisualEncoder(d_model=d_model, num_blocks=num_blocks)
        
        # ✂️ ĐÃ XÓA: self.trajectory_branch
        
        # Lớp chiếu chỉ nhận d_model từ Visual thay vì d_model * 2
        self.visual_proj = nn.Linear(d_model, d_model)

        self.text_proj = nn.Sequential(
            nn.Linear(d_text_in, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.id_cls = nn.Sequential(
            nn.Linear(d_model, d_model),     
            nn.BatchNorm1d(d_model),              
            nn.ReLU(),                                 
            nn.Linear(d_model, num_classes)  
        )

        # ==========================================
        # 2. CROSS-BATCH MEMORY QUEUE (XBM)
        # ==========================================
        self.register_buffer("image_queue", torch.randn(d_model, queue_size))
        self.register_buffer("text_queue", torch.randn(d_model, queue_size))
        
        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        batch_size = image_feat.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.queue_size:
            ptr = 0 
            
        self.image_queue[:, ptr:ptr + batch_size] = image_feat.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feat.T
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    # ✂️ ĐÃ XÓA tham số bbox_features
    def encode_image(self, crop_frames, original_frames):
        if crop_frames.size(2) == 3: 
            crop_frames = crop_frames.permute(0, 2, 1, 3, 4).contiguous()
            original_frames = original_frames.permute(0, 2, 1, 3, 4).contiguous()
            
        # 1. Trích xuất đặc trưng hình ảnh thô
        spatial_x, debug_feat_map, spatial_pe, loss_off = self.video_backbone(crop_frames)
        context_x, _, _, _ = self.video_backbone(original_frames) 
        
        # 2. Qua khối Mamba Visual
        mamba_out, context_out_raw = self.visual_encoder(spatial_x, context_x, spatial_pe)
        
        # 3. Gom cụm đặc trưng hình ảnh (Mean Pooling)
        pooled_visual = mamba_out.contiguous().mean(dim=(1, 2, 3)) 
        pooled_context = context_out_raw.contiguous().mean(dim=(1, 2, 3)) 
        
        # 4. Chiếu và Chuẩn hóa (CHỈ CHIẾU MỖI VISUAL, KHÔNG DUNG HỢP TRAJECTORY)
        visual_embeds_projected = self.visual_proj(pooled_visual)
        visual_embeds_normalized = F.normalize(visual_embeds_projected, p=2, dim=-1)
        
        # Context vẫn chuẩn hóa để train Contrastive Loss In-batch
        context_embeds_normalized = F.normalize(pooled_context, p=2, dim=-1) 

        # 5. Re-ID Classification
        id_logits = self.id_cls(visual_embeds_projected) 

        x_2d = rearrange(mamba_out, 'b t h w d -> (b t) d h w')
        drop_feature = F.dropout2d(x_2d, p=0.1, training=True) if self.training else x_2d

        return {
            "mamba_out_grounded": mamba_out, 
            "drop_feature": drop_feature,    
            "id_logits": id_logits,          
            "visual_embeds": visual_embeds_normalized, 
            "context_embeds": context_embeds_normalized, 
            "debug_feat_map": debug_feat_map,
            "loss_off": loss_off
        }

    def encode_text(self, text_embeds):
        t_embed = self.text_proj(text_embeds)
        return F.normalize(t_embed, p=2, dim=-1)

    # ✂️ ĐÃ XÓA tham số bbox_features
    def forward(self, crop_frames, original_frames, text_embeds=None, dt=1.0):
        # Trích xuất nhánh ảnh
        visual_outputs = self.encode_image(crop_frames, original_frames)
        
        # Trích xuất nhánh text & Tính Logits tương đồng
        if text_embeds is not None:
            t_features = self.encode_text(text_embeds)
            v_features = visual_outputs["visual_embeds"]
            
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * v_features @ t_features.T
            logits_per_text = logits_per_image.T
            
            visual_outputs["logits_per_image"] = logits_per_image
            visual_outputs["logits_per_text"] = logits_per_text

        return visual_outputs