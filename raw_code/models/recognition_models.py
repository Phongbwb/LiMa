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
        
        if hasattr(self.cfg, 'RESNET_CHECKPOINT') and self.cfg.RESNET_CHECKPOINT:
            self.backbone = resnet50(weights=None)
            state_dict = torch.load(self.cfg.RESNET_CHECKPOINT, map_location='cpu')
            if "fc.weight" in state_dict: del state_dict["fc.weight"]
            if "fc.bias" in state_dict: del state_dict["fc.bias"]
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.backbone.fc = nn.Identity() 
        
        # Nếu là direction, nối 2 features (2048 từ crop + 2048 từ full image = 4096)
        self.img_in_dim = 4096 if self.cfg.TASK == 'direction' else 2048 

        self.projection = nn.Sequential(
            nn.Linear(self.img_in_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.embed_dim, self.cfg.NUM_CLASSES)
        )

    def forward(self, inputs):
        if self.cfg.TASK == 'direction':
            crop, full_img = inputs
            # Chia sẻ trọng số (Weight sharing): Dùng chung 1 backbone cho cả 2 ảnh
            feat_crop = self.backbone(crop)
            feat_full = self.backbone(full_img)
            # Ghép đặc trưng lại với nhau dọc theo chiều feature
            features = torch.cat((feat_crop, feat_full), dim=1) 
        else:
            # Task bình thường chỉ dùng crop
            features = self.backbone(inputs)
            
        embeds = self.projection(features)
        logits = self.cls_head(embeds)
        return logits