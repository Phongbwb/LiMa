import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

class MicroLocalization(nn.Module):
    def __init__(self, embed_dim, num_parts=5):
        """
        Module soi chi tiết hạt mịn cho xe máy và ô tô.
        Args:
            embed_dim (int): Số chiều đặc trưng từ ST-Mamba.
            num_parts (int): Số lượng điểm chuẩn (Keypoints) cần soi.
        """
        super().__init__()
        self.num_parts = num_parts
        self.embed_dim = embed_dim

        # 1. Nhánh dự đoán tọa độ các điểm chuẩn (Keypoints) và Độ tự tin (Visibility)
        # Đầu ra: [x, y, v] cho mỗi điểm (3 * num_parts)
        self.part_predictor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_parts * 3)
        )

        # 2. Edge-Snapping: Bộ lọc Sobel khả vi (Trainable Sobel) 
        # Khởi tạo trọng số cố định theo ma trận Sobel nhưng cho phép tinh chỉnh nhẹ
        self.sobel_x = nn.Parameter(torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.sobel_y = nn.Parameter(torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))

        # 3. Channel Attention: Khuếch đại đặc trưng hãng xe
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 8, embed_dim, 1),
            nn.Sigmoid()
        )

        # 4. Fusion Layer: Tổng hợp đặc trưng từ các phần
        self.fusion = nn.Linear(embed_dim * num_parts, embed_dim)

    def forward(self, feature_map, global_tokens):
        """
        feature_map: Đặc trưng không gian từ Backbone [B*T, C, H, W]
        global_tokens: Đặc trưng tổng thể từ ST-Mamba [B*T, C]
        """
        bt, c, h, w = feature_map.shape

        # --- Bước 1: Dự đoán vị trí các điểm cần soi (Foveal Glimpse)  ---
        # part_params: [B*T, num_parts * 3]
        part_params = self.part_predictor(global_tokens)
        part_params = part_params.view(bt, self.num_parts, 3)
        
        # Tách tọa độ (x, y) và độ hiển thị (visibility)
        coords = torch.sigmoid(part_params[:, :, :2]) # Chuẩn hóa về [0, 1]
        visibility = torch.sigmoid(part_params[:, :, 2:]) # Độ tự tin [0, 1]

        # --- Bước 2: Differentiable Foveal Glimpse (Part-RoI Align) ---
        # Chuyển coords thành định dạng box cho RoIAlign [x1, y1, x2, y2]
        # Giả sử mỗi vùng soi có kích thước cố định 15% so với ảnh gốc
        box_size = 0.15 
        rois = []
        for i in range(bt):
            for j in range(self.num_parts):
                cx, cy = coords[i, j, 0] * w, coords[i, j, 1] * h
                rois.append(torch.tensor([i, cx-box_size*w/2, cy-box_size*h/2, cx+box_size*w/2, cy+box_size*h/2]))
        
        rois = torch.stack(rois).to(feature_map.device)
        
        # Trích xuất đặc trưng vùng soi (7x7)
        part_features = roi_align(feature_map, rois, output_size=(7, 7), spatial_scale=1.0)
        # Áp dụng trọng số Visibility: Điểm nào bị khuất sẽ bị triệt tiêu đạo hàm
        part_features = part_features.view(bt, self.num_parts, c, 7, 7)
        part_features = part_features * visibility.view(bt, self.num_parts, 1, 1, 1)

        # --- Bước 3: Edge-Snapping & Channel Attention  ---
        # Làm sắc nét cạnh cho từng vùng đặc trưng
        enhanced_parts = []
        for i in range(self.num_parts):
            p_feat = part_features[:, i] # [B*T, C, 7, 7]
            
            # Tính gradient không gian (Cạnh)
            # Dùng trung bình các kênh để tính cạnh thô
            gray_feat = p_feat.mean(dim=1, keepdim=True)
            grad_x = F.conv2d(gray_feat, self.sobel_x, padding=1)
            grad_y = F.conv2d(gray_feat, self.sobel_y, padding=1)
            edge_map = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
            
            # Khuếch đại đặc trưng bằng Attention
            attn = self.channel_attn(p_feat)
            p_feat = p_feat * attn + (p_feat * edge_map) # Kết hợp đặc trưng + Cạnh
            
            # Pooling về vector
            enhanced_parts.append(F.adaptive_avg_pool2d(p_feat, 1).view(bt, c))

        # --- Bước 4: Fusion ---
        # Nối tất cả các phần lại và đưa về số chiều ban đầu
        combined = torch.cat(enhanced_parts, dim=-1)
        out = self.fusion(combined)
        
        return out, coords, visibility