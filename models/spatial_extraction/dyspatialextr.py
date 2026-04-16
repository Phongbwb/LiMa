import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import math

# ==========================================
# 1. Deformable Patch Embedding (Giữ nguyên - Rất tối ưu)
# ==========================================
class DeformablePatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=4):
        super().__init__()
        self.stride = patch_size
        self.kernel_size = patch_size + 3
        self.padding = self.kernel_size // 2

        self.pre_norm = nn.GroupNorm(1, in_channels)

        self.offset_net = nn.Conv2d(
            in_channels,
            2 * self.kernel_size * self.kernel_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        nn.init.constant_(self.offset_net.weight, 0.)
        nn.init.constant_(self.offset_net.bias, 0.)

        self.weight = nn.Parameter(
            torch.Tensor(embed_dim, in_channels, self.kernel_size, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        x = self.pre_norm(x)
        offsets = torch.tanh(self.offset_net(x)) * 2.0
        out = deform_conv2d(
            input=x,
            offset=offsets,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding
        )
        return out


# ==========================================
# 2. Dynamic Spatial Extraction (Đã vá lỗi)
# ==========================================
class DynamicSpatialExtraction(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, patch_size=4, max_frames=32, base_size=14):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_frames = max_frames

        # (1) BỎ DECOUPLER 
        # (Không còn gọi RiderVehicleDecoupling nữa)

        self.patch_embed = DeformablePatchEmbedding(in_channels, embed_dim, patch_size)

        # (2) Depthwise Conv3D [FIX CAUSAL LEAK]
        # Bỏ padding trên trục thời gian T. Định dạng padding cho Conv3D là (T, H, W).
        self.dw_conv3d = nn.Conv3d(
            embed_dim, embed_dim,
            kernel_size=(3, 3, 3), 
            padding=(0, 1, 1), # <-- Chỉ pad cho Không gian (H, W)
            groups=embed_dim
        )

        self.norm = nn.LayerNorm(embed_dim)

        # (3) Spatiotemporal Positional Embedding [FIX OUT OF BOUNDS]
        # Đổi shape sang định dạng dễ interpolate: [1, Channels, Length] và [1, Channels, H, W]
        self.temp_embed = nn.Parameter(torch.zeros(1, embed_dim, max_frames))
        self.spatial_embed = nn.Parameter(torch.zeros(1, embed_dim, base_size, base_size))
        
        nn.init.normal_(self.temp_embed, std=0.02)
        nn.init.normal_(self.spatial_embed, std=0.02)

    def forward(self, x):
        """
        Input x: [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape

        # (1) 2D processing (Deformable Conv)
        x_2d = x.transpose(1, 2).reshape(B * T, C, H, W)
        
        # feature_map_raw: [B*T, E, H_p, W_p]
        feature_map_raw = self.patch_embed(x_2d)  
        _, E, H_p, W_p = feature_map_raw.shape

        # (2) Restore time để xử lý 3D
        # shape: [B, E, T, H_p, W_p]
        patches_3d = feature_map_raw.view(B, T, E, H_p, W_p).transpose(1, 2)

        # (3) Causal Padding [FIX CAUSAL LEAK]
        # F.pad cho 3D tensor nhận tham số từ chiều cuối lên đầu: (W_L, W_R, H_T, H_B, T_past, T_future)
        # Pad 2 frame vào quá khứ để kernel_size=3 có thể nhìn (t-2, t-1, t).
        patches_pad = F.pad(patches_3d, (0, 0, 0, 0, 2, 0))
        
        features = F.gelu(self.dw_conv3d(patches_pad)) + patches_3d

        # [FIX NÚT THẮT CỔ CHAI]: Đã gỡ F.adaptive_avg_pool3d
        # Mamba chạy cực mượt ở token length lớn (O(N)), nên ta giữ nguyên 
        # độ phân giải H_p x W_p để MicroLocalization có chi tiết.

        # (4) Nội suy Positional Embedding [FIX OUT OF BOUNDS]
        # -- Temporal PE --
        if T != self.max_frames:
            # Nếu số frame T khác max_frames, nội suy 1D (Kéo giãn/Co lại)
            temp_pe = F.interpolate(self.temp_embed, size=T, mode='linear', align_corners=False)
        else:
            temp_pe = self.temp_embed
            
        # Reshape về [1, T, 1, 1, E] để chuẩn bị cộng broadcast
        temp_pe = temp_pe.transpose(1, 2).view(1, T, 1, 1, E)

        # -- Spatial PE --
        if H_p != self.spatial_embed.shape[2] or W_p != self.spatial_embed.shape[3]:
            # Nếu kích thước Patch thay đổi, nội suy 2D
            spatial_pe = F.interpolate(self.spatial_embed, size=(H_p, W_p), mode='bilinear', align_corners=False)
        else:
            spatial_pe = self.spatial_embed
            
        # Reshape về [1, 1, H_p, W_p, E]
        spatial_pe = spatial_pe.permute(0, 2, 3, 1).view(1, 1, H_p, W_p, E)

        # (5) Cộng dồn Positional Features
        # features đang là [B, E, T, H_p, W_p] -> Đưa E ra cuối [B, T, H_p, W_p, E]
        features_transposed = features.permute(0, 2, 3, 4, 1)
        
        # Pytorch tự động broadcast shape khớp với nhau
        pos_features = features_transposed + temp_pe + spatial_pe

        # (6) Flatten → tokens & Normalization
        seq_len = T * H_p * W_p
        tokens = pos_features.reshape(B, seq_len, E)
        tokens = self.norm(tokens)

        # Trả về mask = None vì đã gỡ decoupler. 
        # (Điều này đảm bảo hàm unpack ở LiMaVLM "tokens, mask, feature_map = self.spatial_extractor(video)" không bị lỗi)
        mask = None 

        return tokens, mask, feature_map_raw

class CustomVideoBackbone(nn.Module):
    def __init__(self, d_model=256, num_frames=8, img_size=256):
        super().__init__()
        
        # Thiết lập patch_size sao cho lưới đầu ra là 16x16 (img_size / 16 = 16)
        self.patch_size = 16
        
        # Gọi module của bạn, đảm bảo embed_dim khớp hoàn toàn với d_model của ST-Mamba
        self.extractor = DynamicSpatialExtraction(
            in_channels=3, 
            embed_dim=d_model,  # Phải là 256
            patch_size=self.patch_size, 
            max_frames=num_frames,
            base_size=img_size // self.patch_size # 256 // 16 = 16
        )

    def forward(self, video):
        """
        Input từ DataLoader: video shape [B, T, C, H, W]
        """
        # Module của bạn yêu cầu [B, C, T, H, W]
        video_input = video.transpose(1, 2)
        
        # Trích xuất đặc trưng
        # tokens shape: [B, T * H_p * W_p, d_model]
        tokens, _, _ = self.extractor(video_input)
        
        B, seq_len, D = tokens.shape
        T = video.shape[1]
        
        # Tính toán lại kích thước lưới không gian H_p và W_p
        spatial_pixels = seq_len // T
        H_p = int(math.sqrt(spatial_pixels))
        W_p = H_p
        
        # Đưa về chuẩn [B, T, H_p, W_p, d_model] cho ST-Mamba
        features = tokens.view(B, T, H_p, W_p, D)
        
        return features