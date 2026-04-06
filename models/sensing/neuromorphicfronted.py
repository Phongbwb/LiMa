import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuromorphicFrontend(nn.Module):
    def __init__(self, in_channels=3, h=112, w=112):
        super().__init__()
        # 1. Entorhinal Grid Cells: Ma trận không gian tĩnh 
        # Khởi tạo một mask có thể học để dập tắt 100% báo động giả ở vùng phi vật lý 
        self.grid_mask = nn.Parameter(torch.ones(1, 1, 1, h, w))
        
        # 2. Predictive Coding: Dự đoán khung hình tương lai 
        # Sử dụng Conv3D siêu nhẹ để học quy luật chuyển động thô
        self.predictor = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, in_channels, kernel_size=3, padding=1)
        )
        
        # 3. Saccadic Masking: Cảm biến đạo hàm dht/dt 
        # Ngắt trích xuất đặc trưng nếu video bị nhòe (Motion Blur) quá mức 
        self.blur_threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        Input x: [B, C, T, H, W] (Tensor video thô) [cite: 35]
        """
        b, c, t, h, w = x.shape
        
        # --- BƯỚC 1: Áp dụng Entorhinal Grid Cells ---
        # Nhân với mặt nạ không gian để loại bỏ bầu trời, mái nhà 
        x = x * torch.sigmoid(self.grid_mask)
        
        # --- BƯỚC 2: Saccadic Masking (Lọc nhòe) ---
        # Tính đạo hàm theo thời gian để đo độ biến động đột ngột 
        # shift x theo trục T để tính hiệu số giữa các khung hình
        diff = torch.abs(x[:, :, 1:] - x[:, :, :-1])
        motion_intensity = diff.mean(dim=[1, 2, 3, 4]) # Trung bình độ biến động 
        
        # Tạo mask nhịp sinh học: Nếu motion quá lớn (nhòe), giá trị trả về 0 
        saccadic_gate = torch.sigmoid(self.blur_threshold - motion_intensity)
        saccadic_gate = saccadic_gate.view(b, 1, 1, 1, 1)
        
        # --- BƯỚC 3: Predictive Coding (Lọc vật thể động) ---
        # Dự đoán khung hình t từ các khung hình trước 
        x_pred = self.predictor(x)
        # Chỉ những vật thể có sai số dự đoán lớn (đang chuyển động) mới được giữ lại 
        dynamic_error = torch.abs(x - x_pred)
        
        # Kết hợp: (Lọc chuyển động) x (Chống nhòe) [cite: 17, 31]
        # Đầu ra là video đã được "làm sạch", chỉ còn các phương tiện rõ nét 
        output = dynamic_error * saccadic_gate
        
        return output