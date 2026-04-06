class SaccadicMasking(nn.Module):
    def __init__(self, threshold=0.8):
        super().__init__()
        self.threshold = threshold

    def forward(self, current_state, previous_state):
        # Tính đạo hàm dh_t/dt (sự thay đổi trạng thái giữa 2 khung hình)
        dh_dt = torch.abs(current_state - previous_state)
        velocity_magnitude = torch.mean(dh_dt)
        
        # Nếu xe chạy qua quá nhanh gây nhòe
        if velocity_magnitude > self.threshold:
            # Trả về None để ngắt trích xuất, không lưu vào Vector DB
            return None 
        else:
            return current_state