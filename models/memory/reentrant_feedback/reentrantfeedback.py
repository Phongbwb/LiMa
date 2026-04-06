import torch
import torch.nn as nn

class CorticalReentrantFeedback(nn.Module):
    """
    Tạo 'Tia kỳ vọng' dưới dạng mặt nạ Gaussian 2D để thu hút Bounding Box
    của khung hình hiện tại về vị trí đã được dự đoán từ khung hình trước.
    """
    def __init__(self, feature_dim, spatial_size=(32, 32)):
        super().__init__()
        self.spatial_size = spatial_size
        
        # Mạng hòa trộn (Blend) giữa Đặc trưng hình ảnh nguyên bản và Tia kỳ vọng
        self.feedback_blend = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.SiLU()
        )

    def _generate_gaussian_expectation_ray(self, predicted_bboxes, H, W, device):
        """
        Tạo mặt nạ Gaussian (Tia kỳ vọng) dựa trên tọa độ bốc từ khung hình trước.
        predicted_bboxes: [Batch, 4] định dạng [x_center, y_center, width, height]
        (Lưu ý: Tọa độ này phải được chuẩn hóa về kích thước của Feature Map HxW)
        """
        B = predicted_bboxes.shape[0]
        
        # Tạo lưới tọa độ (Grid)
        y = torch.arange(0, H, device=device).view(1, H, 1).expand(B, H, W).float()
        x = torch.arange(0, W, device=device).view(1, 1, W).expand(B, H, W).float()
        
        # Tách tọa độ dự đoán
        x_c = predicted_bboxes[:, 0].view(B, 1, 1)
        y_c = predicted_bboxes[:, 1].view(B, 1, 1)
        w = predicted_bboxes[:, 2].view(B, 1, 1) + 1e-6 # Tránh chia cho 0
        h = predicted_bboxes[:, 3].view(B, 1, 1) + 1e-6
        
        # Tính toán mặt nạ Gaussian 2D: Lõi sáng nhất ở tâm xe và mờ dần ra viền
        # Công thức: exp( - ( (x - xc)^2 / (w/2)^2 + (y - yc)^2 / (h/2)^2 ) )
        gaussian_mask = torch.exp(
            - ( ((x - x_c) ** 2) / ((w / 2) ** 2) + ((y - y_c) ** 2) / ((h / 2) ** 2) )
        )
        
        # Shape đầu ra: [Batch, 1, H, W]
        return gaussian_mask.unsqueeze(1)

    def forward(self, current_features, predicted_bboxes_from_prev_frame):
        """
        current_features: Đặc trưng hình ảnh hiện tại từ mạng Backbone [B, C, H, W]
        predicted_bboxes_from_prev_frame: Tọa độ kỳ vọng từ khung hình t-1 [B, 4]
        """
        B, C, H, W = current_features.shape
        
        # 1. Khởi tạo Tia kỳ vọng
        expectation_ray = self._generate_gaussian_expectation_ray(
            predicted_bboxes_from_prev_frame, H, W, current_features.device
        )
        
        # 2. Phóng tia kỳ vọng vào đặc trưng hiện tại
        # Những pixel nằm trong vùng tia kỳ vọng sẽ được giữ nguyên hoặc khuếch đại,
        # những pixel nằm ngoài vùng kỳ vọng sẽ bị giảm cường độ.
        guided_features = current_features * expectation_ray
        
        # 3. Hòa trộn mềm mại (Soft Blend) để mạng không bị phụ thuộc 100% vào quá khứ
        # (Tránh trường hợp xe phanh gấp nhưng Bounding box vẫn trôi theo quán tính)
        out_features = current_features + self.feedback_blend(guided_features)
        
        return out_features