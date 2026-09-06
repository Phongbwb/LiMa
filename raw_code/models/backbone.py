import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.ops import deform_conv2d
import torchvision.models as models
import math

# ==========================================
# 1. HÀM TRỰC QUAN HÓA FEATURE MAP
# ==========================================
def visualize_feature_map(feature_tensor, orig_image=None, save_path="feature_map.png", colormap='jet', alpha=0.5):
    """
    Trực quan hóa Feature Map từ Backbone.
    feature_tensor shape mong đợi: [B*T, C, H, W] hoặc [C, H, W]
    """
    if feature_tensor.dim() == 4:
        feature_tensor = feature_tensor[0] 
    
    feature_tensor = feature_tensor.detach().cpu()

    heatmap_2d = torch.mean(feature_tensor, dim=0) 
    
    hm_min, hm_max = heatmap_2d.min(), heatmap_2d.max()
    heatmap_norm = (heatmap_2d - hm_min) / (hm_max - hm_min + 1e-8)
    heatmap_np = heatmap_norm.numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    
    if orig_image is not None:
        if isinstance(orig_image, Image.Image):
            img_w, img_h = orig_image.size
            ax.imshow(orig_image)
        elif isinstance(orig_image, np.ndarray):
            img_h, img_w = orig_image.shape[:2]
            ax.imshow(orig_image)

        heatmap_tensor = torch.from_numpy(heatmap_np).unsqueeze(0).unsqueeze(0)
        heatmap_resized = F.interpolate(heatmap_tensor, size=(img_h, img_w), mode='bicubic', align_corners=False).squeeze().numpy()
        
        cmap = plt.get_cmap(colormap)
        ax.imshow(heatmap_resized, cmap=cmap, alpha=alpha)
    else:
        cmap = plt.get_cmap(colormap)
        im = ax.imshow(heatmap_np, cmap=cmap)
        fig.colorbar(im, ax=ax)

    ax.axis('off')
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ Đã lưu ảnh Feature Map tại: {save_path}")

# ==========================================
# 2. CÁC LỚP PHỤ TRỢ (Đã tối ưu)
# ==========================================
class DeformablePatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=2):
        super().__init__()
        self.stride = patch_size
        
        self.kernel_size = 3 if patch_size == 2 else patch_size
        self.padding = 1 if self.kernel_size == 3 else (self.kernel_size - 1) // 2

        self.post_norm = nn.GroupNorm(1, embed_dim)
        
        # 🔥 KHỞI TẠO ZERO (Zero-Initialization)
        # Giúp ở những epoch đầu, mạng hoàn toàn tin tưởng vào ResNet Shortcut
        nn.init.zeros_(self.post_norm.weight) 

        # 🔥 Thu hẹp vùng nhìn của mạng đoán Offset (kernel_size=1)
        self.offset_mask_net = nn.Conv2d(
            in_channels, 
            3 * self.kernel_size ** 2, 
            kernel_size=1, 
            padding=0, 
            stride=self.stride
        )
        nn.init.zeros_(self.offset_mask_net.weight)
        nn.init.zeros_(self.offset_mask_net.bias)
        
        # 🔥 Khởi tạo scale offset nhỏ để ban đầu giống CNN thuần
        self.offset_scale = nn.Parameter(torch.tensor(0.5))

        self.weight = nn.Parameter(torch.Tensor(embed_dim, in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        out_offset_mask = self.offset_mask_net(x)
        o1, o2, mask = torch.chunk(out_offset_mask, 3, dim=1)
        
        # 🔥 Kẹp chặt Offset để tránh loang lổ ra mặt đường
        offsets = torch.clamp(torch.cat((o1, o2), dim=1), -1.5, 1.5) * self.offset_scale
        
        out = deform_conv2d(
            x, 
            offsets, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            mask=torch.sigmoid(mask)
        )
        # Trả về thêm offsets để phạt Loss
        return self.post_norm(out), offsets

class SpatialRefinementBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dw_conv2d = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Conv2d(d_model, d_model * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Conv2d(d_model * 4, d_model, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.dw_conv2d(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() 
        return residual + self.pw_conv2(self.act(self.pw_conv1(x)))

class SpatialGating(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.Conv2d(d_model, d_model // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model // 4, d_model, kernel_size=1)
        )
        # 🔥 Thêm Temperature để cắt gắt phần mask nền (giảm nhiễu)
        self.temperature = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, x): 
        mask = torch.sigmoid(self.gate(x) * self.temperature)
        return x*mask 

# ==========================================
# 3. BACKBONE TỔNG THỂ VỚI SHORTCUT
# ==========================================

class CustomVideoBackbone(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        self.early_extractor = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
            resnet.layer1, 
            resnet.layer2  
        )

        # Đóng băng gradient ResNet
        for param in self.early_extractor.parameters():
            param.requires_grad = False

        # 🔥 Lớp Shortcut để đẩy trực tiếp Feature ResNet lên sau Deformable
        self.resnet_shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2), # Giảm H, W xuống 1 nửa khớp với Deformable
            nn.Conv2d(128, d_model, kernel_size=1, bias=False), # Nâng số kênh 128 -> 256
            nn.GroupNorm(1, d_model)
        )

        self.deform_patch_embed = DeformablePatchEmbedding(
            in_channels=128, 
            embed_dim=d_model, 
            patch_size=2 
        )

        self.spatial_gate = SpatialGating(d_model)
        
        self.temp_mlp = nn.Sequential(nn.Linear(9, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.spatial_mlp = nn.Conv2d(18, d_model, 1) 
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(d_model)

    def train(self, mode=True):
        """🔥 Ghi đè hàm train để đảm bảo Batch Norm của ResNet luôn bị khóa cứng"""
        super().train(mode)
        self.early_extractor.eval() 
        return self

    def get_fourier_features(self, coords, num_bands=4):
        features = [coords]
        for freq in [2**i for i in range(num_bands)]:
            features.append(torch.sin(math.pi * freq * coords))
            features.append(torch.cos(math.pi * freq * coords))
        return torch.cat(features, dim=-1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_2d = x.transpose(1, 2).reshape(B * T, C, H, W)
        
        # 1. Đặc trưng sắc nét từ ResNet [B*T, 128, H/8, W/8]
        feat_resnet = self.early_extractor(x_2d)
        
        # 2. Đặc trưng biến dạng từ Deformable [B*T, 256, H/16, W/16]
        feat_deform, offsets = self.deform_patch_embed(feat_resnet)
        
        # 3. Đường tắt (Shortcut) cho ResNet [B*T, 256, H/16, W/16]
        shortcut = self.resnet_shortcut(feat_resnet)
        
        # 🔥 4. CỘNG GỘP: Giữ trọn ResNet, bổ sung Deformable
        feat_2d = shortcut
        
        # Lấy bản sao để vẽ heatmap kiểm tra
        feature_to_visualize = feat_2d.clone()

        _, E, Hp, Wp = feat_2d.shape

        # 5. Gating cắt nhiễu
        feat_2d = self.spatial_gate(feat_2d)
        feat_3d = feat_2d.view(B, T, E, Hp, Wp)

        # 6. Positional Encoding
        t = torch.linspace(-1, 1, T, device=x.device).view(T, 1)
        t_fourier = self.get_fourier_features(t) 
        temp_pe = self.temp_mlp(t_fourier).view(1, T, 1, 1, E)

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, Hp, device=x.device),
            torch.linspace(-1, 1, Wp, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0) 
        grid_fourier = self.get_fourier_features(grid).permute(0, 3, 1, 2) 
        spatial_pe = self.spatial_mlp(grid_fourier).permute(0, 2, 3, 1).unsqueeze(1) 

        out = feat_3d.permute(0, 1, 3, 4, 2) + (temp_pe + spatial_pe) * self.pe_scale
        out = self.norm(out)
        
        # 🔥 Trả về chuẩn bị cho Mamba và Loss
        return out.contiguous(), feature_to_visualize, spatial_pe, offsets