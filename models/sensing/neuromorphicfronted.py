import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuromorphicFrontend(nn.Module):
    def __init__(self, in_channels=3, base_h=112, base_w=112):
        super().__init__()

        # =========================
        # (1) Grid Mask (Linh hoạt kích thước)
        # =========================
        self.grid_mask = nn.Parameter(torch.zeros(1, 1, 1, base_h, base_w))

        # =========================
        # (2) Lightweight Predictive Coding
        # =========================
        self.predictor = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        )

        # =========================
        # (3) Learnable motion threshold (Sửa tên cho đúng bản chất)
        # =========================
        self.motion_threshold = nn.Parameter(torch.tensor(0.1)) # Giá trị khởi tạo nhỏ

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        Returns:
            output: [B, C, T, H, W]
            pred_loss: L1 Loss để train predictor (chỉ dùng lúc training)
        """
        B, C, T, H, W = x.shape

        # =========================
        # (1) Grid Mask (Xử lý linh hoạt hình học)
        # =========================
        # Ép tensor 5D [1, 1, 1, H, W] về 4D [1, 1, H, W] bằng cách bỏ trục T
        mask_4d = self.grid_mask.squeeze(2)
        
        # Nội suy trên 4D
        mask_4d = F.interpolate(
            mask_4d, size=(H, W), mode='bilinear', align_corners=False
        )
        
        # Trả lại trục T để thành 5D [1, 1, 1, H, W] khớp với video
        mask = mask_4d.unsqueeze(2)
        
        mask = torch.sigmoid(mask) * 0.5 + 0.5
        x = x * mask

        # =========================
        # (2) Predictive Coding
        # =========================
        x_prev = x[:, :, :-1]   # [B, C, T-1, H, W]
        x_next = x[:, :, 1:]    # [B, C, T-1, H, W]

        x_pred = self.predictor(x_prev)
        
        # [QUAN TRỌNG]: Tính Auxiliary Loss cho Predictor
        pred_loss = F.l1_loss(x_pred, x_next) if self.training else 0.0

        dynamic_error = torch.abs(x_pred - x_next)
        dynamic_error = dynamic_error / (dynamic_error.mean(dim=[2, 3, 4], keepdim=True) + 1e-6)

        # =========================
        # (3) Motion Map
        # =========================
        diff = torch.abs(x_next - x_prev)
        motion_map = diff.mean(dim=1, keepdim=True)  # [B, 1, T-1, H, W]

        # =========================
        # (4) Saccadic Gate (FIX LOGIC)
        # =========================
        # Highlight motion: Chuyển động vượt ngưỡng -> Gate mở (tiến về 1)
        saccadic_gate = torch.sigmoid((motion_map - self.motion_threshold) * 10.0) # Nhân 10 để tạo độ gắt (sharpness) cho cổng

        # =========================
        # (5) Combine & Temporal Padding (FIX)
        # =========================
        combined_feat = (0.5 * x_next) + (0.5 * dynamic_error)
        combined_feat = combined_feat * saccadic_gate

        # [FIX]: Thay vì đệm 0 làm đứt gãy thông tin, đệm copy (replicate) frame đầu tiên
        # Do ta đã mất 1 frame ở quá trình x_next, nên cần bù 1 frame vào đầu (thời điểm t=0)
        # Thời điểm t=0 chưa có motion, ta truyền nguyên bản x[:, :, 0:1] vào
        first_frame_feat = x[:, :, 0:1] 
        
        output = torch.cat([first_frame_feat, combined_feat], dim=2) # [B, C, T, H, W]

        return output, pred_loss