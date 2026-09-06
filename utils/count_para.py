
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lima import LiMaVLM
from models.classifier_trajectory import IntentPredictor
def count_parameters(model):
    # Tính tổng tất cả tham số
    total_params = sum(p.numel() for p in model.parameters())
    
    # Tính tổng các tham số có bật requires_grad=True
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*40)
    print(f"Tổng số tham số (Total):       {total_params:,}")
    print(f"Tham số huấn luyện (Trainable):  {trainable_params:,}")
    print(f"Tham số đóng băng (Frozen):      {total_params - trainable_params:,}")
    print("="*40)
    
    return total_params, trainable_params

# Khởi tạo mô hình của bạn (với các tham số mặc định)
# Lưu ý: Cần đảm bảo các class CustomVideoBackbone, TwoStreamVisualEncoder... đã được load.
model = IntentPredictor()

# Gọi hàm đo
count_parameters(model)