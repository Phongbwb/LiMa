import torch
import torch.nn as nn

class PredictiveCodingGate(nn.Module):
    """
    Mô phỏng cơ chế Predictive Coding: Mạng dự đoán bối cảnh tĩnh (khung hình tương lai)
    và chỉ cho phép các "sai số dự đoán" (vật thể chuyển động) đi qua lõi Mamba.
    """
    def __init__(self, alpha=0.01, threshold=0.1):
        super().__init__()
        # Hệ số alpha: Tốc độ cập nhật nền. 
        # Alpha nhỏ (0.01) giúp mạng không vô tình "xóa" mất các xe đang dừng đèn đỏ
        self.alpha = alpha 
        
        # Ngưỡng (Threshold) để xác định sự thay đổi (0.0 đến 1.0)
        self.threshold = threshold 
        
        # Biến lưu trữ bối cảnh nền (Sẽ được đẩy lên GPU tự động)
        self.background_model = None

    def forward(self, current_frame):
        """
        current_frame: Tensor hình ảnh hiện tại, shape (Batch, Channels, Height, Width)
                       Giá trị pixel đã được chuẩn hóa về [0, 1]
        """
        # 1. Khởi tạo mô hình nền nếu đây là khung hình đầu tiên của luồng video
        if self.background_model is None:
            self.background_model = current_frame.clone().detach()
            # Ở frame đầu, ta cho phép toàn bộ khung hình đi qua để khởi động
            return current_frame, torch.ones_like(current_frame[:, :1, :, :])

        # 2. Dự đoán: Khung hình "Kỳ vọng" chính là bối cảnh nền tĩnh
        expected_frame = self.background_model

        # 3. Tính toán Sai số dự đoán (Prediction Error)
        prediction_error = torch.abs(current_frame - expected_frame)

        # 4. Tạo mặt nạ chuyển động (Motion Mask)
        # Gộp trung bình sai số của 3 kênh RGB (hoặc trích xuất kênh độ sáng)
        error_gray = prediction_error.mean(dim=1, keepdim=True)
        
        # Tạo mặt nạ nhị phân: 1 nếu là vật thể chuyển động, 0 nếu là nền tĩnh
        motion_mask = (error_gray > self.threshold).float()

        # 5. Cập nhật lại khung hình kỳ vọng (Học hỏi liên tục thích ứng với ánh sáng ngày/đêm)
        # Sử dụng .detach() để ngắt đồ thị tính toán, tiết kiệm VRAM
        self.background_model = (self.alpha * current_frame.clone().detach()) + \
                                ((1.0 - self.alpha) * self.background_model)

        # 6. GÁC CỔNG (Gating): Xóa sổ các vùng tĩnh
        # Phép nhân element-wise biến toàn bộ đường nhựa, tòa nhà, cây cối thành các pixel đen (giá trị 0)
        active_features = current_frame * motion_mask

        # Trả về các pixel đang chuyển động và mặt nạ để theo dõi
        return active_features, motion_mask