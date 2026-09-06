import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter
from tqdm import tqdm

# Import chuẩn cho tính năng Mixed Precision (AMP)
from torch.cuda.amp import autocast

# Import Mô hình Đơn nhiệm Tối ưu
from models.classifier_mamba import PureColorMambaClassifier

# ==========================================
# 1. BỘ DỮ LIỆU INFERENCE (ĐỒNG BỘ 100% VỚI TRAIN)
# ==========================================
class VehicleTrackDataset(Dataset):
    def __init__(self, json_path, base_dir, transform=None):
        self.transform = transform
        self.base_dir = base_dir 
        self.data = []

        print(f"====> Đang đọc dữ liệu Track Test từ: {json_path}...")
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
            for uuid_key, item in raw_data.items():
                frames = item["frames"]
                boxes = item["boxes"]
                
                assert len(frames) == len(boxes), f"Lỗi ở UUID {uuid_key}: Số lượng frames khác boxes"
                
                for frame_path, box in zip(frames, boxes):
                    img_path = os.path.join(self.base_dir, frame_path)
                    self.data.append({
                        "uuid": uuid_key,
                        "full_path": img_path,
                        "box": box
                    })

        print(f"====> Hoàn tất! Tổng số ảnh (frames) cần dự đoán: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        uuid_key = item["uuid"]
        x, y, w, h = item["box"]
        
        try:
            # ---------------------------------------------------------
            # ĐỒNG BỘ CHUẨN: SỬ DỤNG PIL VÀ CROP 5% PADDING
            # ---------------------------------------------------------
            full_image = Image.open(item["full_path"]).convert("RGB")
            
            # Thêm padding nhẹ 5% để lấy thêm bối cảnh viền xe (GIỐNG HỆT TRAIN)
            padding_x, padding_y = int(w * 0.05), int(h * 0.05)
            crop_image = full_image.crop((
                max(0, x - padding_x), 
                max(0, y - padding_y), 
                min(full_image.width, x + w + padding_x), 
                min(full_image.height, y + h + padding_y)
            )) 

            if self.transform:
                crop_tensor = self.transform(crop_image)
            else:
                crop_tensor = transforms.ToTensor()(crop_image)
                
        except Exception as e:
            # Fallback nếu đường dẫn ảnh lỗi (trả về tensor đen)
            print(f"Lỗi đọc ảnh {item['full_path']}: {e}")
            crop_tensor = torch.zeros((3, 336, 336))
            
        return crop_tensor, uuid_key

# ==========================================
# 2. HÀM INFERENCE CÓ CƠ CHẾ VOTING
# ==========================================
def infer_and_vote():
    BASE_DIR = 'data/cityflownl/data' 
    TEST_JSON_PATH = './data/json/test-tracks.json' 
    WEIGHTS_PATH = './checkpoints/recognition/mamba_color_epoch3.pth' # Trỏ tới model tốt nhất của bạn
    OUTPUT_JSON_PATH = './data/json/test_tracks_color.json'
    
    NUM_CLASSES = 8 # Phải khớp với config lúc train
    EMBED_DIM = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n====> BẮT ĐẦU INFERENCE MÀU SẮC TRÊN {device}")

    # =========================================================
    # ĐỒNG BỘ TRANSFORMS 
    # =========================================================
    test_transforms = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.CenterCrop((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VehicleTrackDataset(TEST_JSON_PATH, base_dir=BASE_DIR, transform=test_transforms)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Khởi tạo mô hình
    model = PureColorMambaClassifier(
        num_colors=NUM_CLASSES, 
        d_model=EMBED_DIM
    ).to(device)
    
    print(f"====> Đang tải weights từ: {WEIGHTS_PATH}")
    # Load model an toàn
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    model.eval()

    uuid_predictions = {}

    print("\n====> Đang chạy mô hình dự đoán từng frame...")
    with torch.no_grad():
        # Dùng tqdm để hiển thị thanh tiến trình
        for inputs, uuid_keys in tqdm(dataloader, desc="Dự đoán Batch"):
            inputs = inputs.to(device)
            
            # Sử dụng autocast để tăng tốc và đồng bộ dtype với lúc train
            with autocast():
                logits = model(inputs)
            
            _, preds = torch.max(logits, dim=1)
            preds = preds.cpu().numpy().tolist()
            
            # Gom nhóm dự đoán theo UUID
            for uuid, pred_id in zip(uuid_keys, preds):
                if uuid not in uuid_predictions:
                    uuid_predictions[uuid] = []
                uuid_predictions[uuid].append(pred_id)

    # ==========================================
    # 4. MAJORITY VOTING 
    # ==========================================
    print("\n====> Tiến hành Majority Voting cho từng UUID...")
    final_results = {}
    
    for uuid, list_preds in uuid_predictions.items():
        vote_counts = Counter(list_preds)
        best_id = vote_counts.most_common(1)[0][0] # Lấy nhãn xuất hiện nhiều nhất
        final_results[uuid] = {"id": best_id}

    # ==========================================
    # 5. LƯU FILE JSON
    # ==========================================
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"====> HOÀN TẤT! Đã lưu kết quả tại: {OUTPUT_JSON_PATH}")
    print(f"Tổng số UUID đã dự đoán: {len(final_results)}\n")

if __name__ == '__main__':
    # Tối ưu hóa backend cho quá trình Inference
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True 
        
    infer_and_vote()