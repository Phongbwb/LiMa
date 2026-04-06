import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import math

# ==========================================
# 1. Module Tách người lái (Decoupling Mask)
# ==========================================
class RiderVehicleDecoupling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Sử dụng 3D Conv siêu nhẹ để dự đoán mặt nạ xuyên suốt thời gian
        self.mask_net = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # Ép giá trị về [0, 1]
        )

    def forward(self, x):
        """
        x: [Batch, Channels, Time, Height, Width]
        """
        mask = self.mask_net(x)
        # Lấy hình ảnh gốc TRỪ ĐI phần nhiễu/người lái
        # Những điểm mask tiến về 1 (người) thì (1 - mask) tiến về 0 (xóa sổ)
        clean_x = x * (1.0 - mask)
        return clean_x, mask

# ==========================================
# 2. Nhúng Bản vá Biến dạng (Deformable Patch) - ĐÃ SỬA LỖI
# ==========================================
class DeformablePatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=4):
        super().__init__()
        self.stride = patch_size
        # Nhúng chồng chéo (Overlapping): Kernel lớn hơn Stride
        self.kernel_size = patch_size + 3 
        self.padding = self.kernel_size // 2

        # SỬA LỖI Ở ĐÂY: Thêm stride=self.stride để ép kích thước offset khớp với output
        # Khuyên dùng: kernel_size và padding nên giống hệt với lớp deform_conv chính
        self.offset_net = nn.Conv2d(
            in_channels, 
            2 * self.kernel_size * self.kernel_size, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding
        )
        
        # MẸO QUAN TRỌNG: Khởi tạo offset bằng 0 để mạng ổn định ở các epoch đầu
        nn.init.constant_(self.offset_net.weight, 0.)
        nn.init.constant_(self.offset_net.bias, 0.)
        
        # Trọng số của phép tích chập biến dạng
        self.weight = nn.Parameter(
            torch.Tensor(embed_dim, in_channels, self.kernel_size, self.kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        """
        deform_conv2d chỉ nhận 4D Tensor, nên ta ép Time vào Batch
        x: [B*T, C, H, W]
        """
        # Mạng tự học cách bẻ cong các lưới cắt ảnh bám theo khung xe
        offsets = self.offset_net(x)
        
        # Áp dụng Tích chập Biến dạng
        out = deform_conv2d(
            input=x, 
            offset=offsets, 
            weight=self.weight, 
            stride=self.stride, 
            padding=self.padding
        )
        return out

# ==========================================
# 3. Trích xuất Cục bộ & Lắp ráp Tổng thể
# ==========================================
class DynamicSpatialExtraction(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, patch_size=4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Tách nhiễu động
        self.decoupler = RiderVehicleDecoupling(in_channels)
        
        # 2. Trích xuất Patch uốn lượn
        self.patch_embed = DeformablePatchEmbedding(in_channels, embed_dim, patch_size)
        
        # 3. Bơm Ngữ nghĩa Cục bộ (Depthwise Conv3D)
        # Trộn thông tin giữa các khung hình (Time) và không gian (H, W) sau khi tạo Patch
        self.dw_conv3d = nn.Conv3d(
            in_channels=embed_dim, 
            out_channels=embed_dim, 
            kernel_size=3, 
            padding=1, 
            groups=embed_dim # Khóa quan trọng: groups=embed_dim tạo ra Depthwise Conv, tiết kiệm 90% VRAM
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Luồng xuôi toàn trình
        Input: Tensor Video thô [Batch, Channels, Time, Height, Width]
        Output: Tensor 1D phẳng chuẩn bị đưa vào ST-Mamba [Batch, SeqLen, EmbedDim]
        """
        B, C, T, H, W = x.shape

        # Bước 1: Xóa người lái / nhiễu
        x_clean, mask = self.decoupler(x)

        # Bước 2: Ép chiều Time vào Batch để xử lý 2D Deformable Conv
        # Từ [B, C, T, H, W] -> [B, T, C, H, W] -> [B*T, C, H, W]
        x_clean_2d = x_clean.transpose(1, 2).reshape(B * T, C, H, W)
        
        # Tạo Patch
        patches = self.patch_embed(x_clean_2d) # [B*T, EmbedDim, H', W']
        _, E, H_prime, W_prime = patches.shape

        # Bước 3: Phục hồi lại chiều Time để chạy Depthwise Conv3D
        # Từ [B*T, E, H', W'] -> [B, T, E, H', W'] -> [B, E, T, H', W']
        patches_3d = patches.reshape(B, T, E, H_prime, W_prime).transpose(1, 2)
        
        # Trộn đặc trưng cục bộ (Local Semantic Booster)
        features = F.gelu(self.dw_conv3d(patches_3d))

        # Bước 4: Trải phẳng (Flatten) thành chuỗi 1D cho ST-Mamba
        # [B, E, T, H', W'] -> [B, E, T * H' * W'] -> [B, T * H' * W', E]
        seq_len = T * H_prime * W_prime
        tokens = features.view(B, E, seq_len).transpose(1, 2)
        
        # Chuẩn hóa chuẩn bị vào Mamba
        tokens = self.norm(tokens)

        return tokens, mask