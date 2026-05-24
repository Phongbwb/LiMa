import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50_SingleTask(nn.Module):
    def __init__(self, model_cfg):
        super(ResNet50_SingleTask, self).__init__()
        self.cfg = model_cfg
        self.embed_dim = self.cfg.EMBED_DIM
        
        print(f"====> Khởi tạo ResNet50 cho tác vụ: {self.cfg.TASK.upper()}")
        
        # 1. Khởi tạo Backbone ResNet50
        if hasattr(self.cfg, 'RESNET_CHECKPOINT') and self.cfg.RESNET_CHECKPOINT:
            self.backbone = resnet50(weights=None)
            state_dict = torch.load(self.cfg.RESNET_CHECKPOINT, map_location='cpu')
            if "fc.weight" in state_dict: del state_dict["fc.weight"]
            if "fc.bias" in state_dict: del state_dict["fc.bias"]
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.backbone.fc = nn.Identity() 
        self.img_in_dim = 2048 

        # 2. Lớp Projection 
        self.projection = nn.Sequential(
            nn.Linear(self.img_in_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU()
        )

        # 3. Lớp Phân loại (Classification Head)
        self.cls_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.cfg.NUM_CLASSES)
        )

    def forward(self, crop):
        # Trích xuất đặc trưng
        features = self.backbone(crop)
        embeds = self.projection(features)
        
        # Có thể tắt L2 Normalize nếu dùng CrossEntropyLoss thông thường để hội tụ nhanh hơn
        # embeds = F.normalize(embeds, p=2, dim=-1) 
        
        logits = self.cls_head(embeds)
        return logits