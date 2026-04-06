import torch
import torch.nn as nn

class EntorhinalGridCells(nn.Module):
    """
    Ma trận Không gian tĩnh mô phỏng Tế bào lưới.
    Lọc bỏ hoàn toàn các khu vực phi vật lý (trời, mái nhà) để chống báo động giả.
    """
    def __init__(self, height, width):
        super().__init__()
        # Khởi tạo ma trận không gian: Mặc định ban đầu cho phép mọi khu vực (toàn số 1)
        self.register_buffer('spatial_map', torch.ones(1, 1, height, width))
        
        # Bộ tích lũy chuyển động dùng cho giai đoạn Auto-Calibration (Hiệu chuẩn)
        self.register_buffer('motion_accumulator', torch.zeros(1, 1, height, width))
        self.is_calibrated = False

    def accumulate_motion(self, motion_mask):
        """
        Nhận mặt nạ chuyển động từ Predictive Coding để tích lũy.
        (Chạy trong khoảng 1-2 tiếng đầu tiên khi lắp camera)
        """
        if not self.is_calibrated:
            self.motion_accumulator += motion_mask.sum(dim=0, keepdim=True)

    def finalize_calibration(self, threshold_ratio=0.05):
        """
        Chốt "Ma trận Không gian tĩnh" sau khi thu thập đủ dữ liệu.
        Vùng nào hiếm khi hoặc không bao giờ có chuyển động (dưới ngưỡng) sẽ bị biến thành 0.
        """
        max_motion = self.motion_accumulator.max()
        threshold = max_motion * threshold_ratio
        
        # Cập nhật spatial_map: 1 (Hợp lệ / Đường đi), 0 (Phi vật lý / Bầu trời, mái nhà)
        self.spatial_map = (self.motion_accumulator > threshold).float()
        self.is_calibrated = True
        
        # Giải phóng bộ nhớ của bộ tích lũy
        self.motion_accumulator = torch.zeros_like(self.motion_accumulator)

    def manual_roi_setup(self, polygon_mask):
        """
        Cho phép người dùng tự vẽ vùng Region of Interest (ROI) nếu không muốn chờ Auto-Calibration.
        """
        self.spatial_map.data = polygon_mask.float().unsqueeze(0).unsqueeze(0).to(self.spatial_map.device)
        self.is_calibrated = True

    def forward(self, features):
        """
        Áp dụng Ma trận Không gian tĩnh lên đặc trưng hình ảnh.
        features: Shape (Batch, Channels, Height, Width)
        """
        # Phép nhân element-wise: Tất cả các pixel rơi vào vùng 0 (bầu trời, mái nhà)
        # sẽ bị dập tắt (nhân với 0), bất kể ở đó có chuyển động hay không.
        filtered_features = features * self.spatial_map
        return filtered_features
    
class SensingAndSpatialFiltering(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        # 1. Dự đoán chuyển động
        self.predictive_coding = PredictiveCodingGate(alpha=0.01, threshold=0.1)
        # 2. Ma trận không gian tĩnh
        self.grid_cells = EntorhinalGridCells(height, width)

    def forward(self, current_frame):
        # Bước 1: Trích xuất các pixel đang chuyển động (Bỏ qua nền tĩnh như mặt đường)
        active_features, motion_mask = self.predictive_coding(current_frame)
        
        # (Tùy chọn) Trong vài giờ đầu tiên, hệ thống âm thầm học địa hình
        if self.training and not self.grid_cells.is_calibrated:
            self.grid_cells.accumulate_motion(motion_mask)
            
        # Bước 2: Dập tắt báo động giả (Bỏ qua chim bay, mây trôi ở bầu trời/mái nhà)
        final_filtered_features = self.grid_cells(active_features)
        
        return final_filtered_features