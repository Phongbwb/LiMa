import torch

MODEL_WEIGHTS = "./checkpoints/intent/intent_ep1.pth"
print(f"Đang siêu âm file: {MODEL_WEIGHTS}")

try:
    state_dict = torch.load(MODEL_WEIGHTS, map_location="cpu")
    
    nan_count = 0
    zero_count = 0
    total_tensors = 0
    
    for name, tensor in state_dict.items():
        total_tensors += 1
        if torch.isnan(tensor).any():
            nan_count += 1
            print(f"[CẢNH BÁO] Layer '{name}' chứa giá trị NaN!")
        elif torch.sum(torch.abs(tensor)).item() == 0:
            zero_count += 1
            print(f"[CẢNH BÁO] Layer '{name}' toàn số 0!")
            
    print("-" * 30)
    print(f"Tổng số Layers: {total_tensors}")
    print(f"Số Layer bị NaN: {nan_count}")
    print(f"Số Layer bị Zero: {zero_count}")
    
    if nan_count > 0 or zero_count > 0:
        print("\n=> KẾT LUẬN: File weights đã bị hỏng trong lúc Train. Bạn phải Train lại!")
    else:
        print("\n=> KẾT LUẬN: File weights bình thường, các ma trận đều có giá trị.")

except Exception as e:
    print(f"Lỗi không thể đọc file: {e}")