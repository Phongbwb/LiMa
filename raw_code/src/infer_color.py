import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter

# Import Mô hình Đơn nhiệm Tối ưu
from models.classifier_mamba import PureColorMambaClassifier

# ==========================================
# 1. BỘ DỮ LIỆU INFERENCE (ĐỒNG BỘ VỚI TRAIN)
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
            # 🔥 ĐỒNG BỘ: SỬ DỤNG OPENCV & CLAHE NHƯ FILE TRAIN
            # ---------------------------------------------------------
            img_bgr = cv2.imread(item["full_path"])
            if img_bgr is None:
                raise ValueError
                
            crop_bgr = img_bgr[y:y+h, x:x+w]

            lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            
            limg = cv2.merge((cl, a, b))
            crop_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB) 
            
            img_pil = Image.fromarray(crop_enhanced)

        except Exception as e:
            # Fallback nếu đường dẫn ảnh lỗi
            img_pil = Image.new('RGB', (336, 336), (0, 0, 0))

        if self.transform:
            crop_tensor = self.transform(img_pil)
        else:
            crop_tensor = transforms.ToTensor()(img_pil)
            
        return crop_tensor, uuid_key

# ==========================================
# 2. HÀM INFERENCE CÓ CƠ CHẾ VOTING (AUTO LOOP 1-30)
# ==========================================
def infer_and_vote():
    BASE_DIR = 'data/cityflownl/data' 
    TEST_JSON_PATH = './data/json/test-tracks.json' 
    CHECKPOINT_DIR = './checkpoints/recognition'
    OUTPUT_DIR = './data/json'
    
    NUM_CLASSES = 8 # Phải khớp với config lúc train
    EMBED_DIM = 256
    NUM_MAMBA_BLOCKS = 6 # Khớp với kiến trúc 6 lớp Mamba

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n====> BẮT ĐẦU INFERENCE MÀU SẮC TRÊN {device}")

    # =========================================================
    # 🔥 ĐỒNG BỘ TRANSFORMS (BAO GỒM NORMALIZE)
    # =========================================================
    test_transforms = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.CenterCrop((336, 336)),
        transforms.ToTensor(),
        # Bắt buộc phải Normalize vì mô hình Train đã học với phổ dữ liệu này
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Khởi tạo Dataset và DataLoader 1 lần duy nhất ở ngoài vòng lặp
    dataset = VehicleTrackDataset(TEST_JSON_PATH, base_dir=BASE_DIR, transform=test_transforms)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Khởi tạo mô hình 1 lần duy nhất ở ngoài vòng lặp
    model = PureColorMambaClassifier(
        num_colors=NUM_CLASSES, 
        d_model=EMBED_DIM, 
        num_mamba_blocks=NUM_MAMBA_BLOCKS
    ).to(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # 🔥 VÒNG LẶP SUY LUẬN TỪ EPOCH 1 ĐẾN 30
    # ==========================================
    for epoch in range(1, 31):
        weights_path = os.path.join(CHECKPOINT_DIR, f'mamba_color_ep{epoch}.pth')
        output_json_path = os.path.join(OUTPUT_DIR, f'test_tracks_color_ep{epoch}.json')

        # Bỏ qua nếu file weights của epoch đó chưa tồn tại
        if not os.path.exists(weights_path):
            print(f"\n[!] Bỏ qua Epoch {epoch}: Không tìm thấy {weights_path}")
            continue

        print(f"\n[{'-'*40}]")
        print(f"====> ĐANG XỬ LÝ EPOCH {epoch}")
        print(f"====> Đang tải weights từ: {weights_path}")
        
        # Nạp trọng số mới và chuyển sang chế độ đánh giá
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()

        # BẮT BUỘC: Xóa trắng từ điển dự đoán của epoch trước đó
        uuid_predictions = {}

        print("  + Đang chạy mô hình dự đoán từng frame...")
        with torch.no_grad():
            for i, (inputs, uuid_keys) in enumerate(dataloader):
                inputs = inputs.to(device)
                logits = model(inputs)
                
                _, preds = torch.max(logits, dim=1)
                preds = preds.cpu().numpy().tolist()
                
                for uuid, pred_id in zip(uuid_keys, preds):
                    if uuid not in uuid_predictions:
                        uuid_predictions[uuid] = []
                    uuid_predictions[uuid].append(pred_id)
                    
                if (i + 1) % 10 == 0:
                    print(f"    - Đã xử lý Batch [{i+1}/{len(dataloader)}]")

        # ==========================================
        # 4. MAJORITY VOTING CHO EPOCH HIỆN TẠI
        # ==========================================
        final_results = {}
        for uuid, list_preds in uuid_predictions.items():
            vote_counts = Counter(list_preds)
            best_id = vote_counts.most_common(1)[0][0]
            final_results[uuid] = {"id": best_id}

        # ==========================================
        # 5. LƯU FILE JSON CHO EPOCH HIỆN TẠI
        # ==========================================
        with open(output_json_path, 'w') as f:
            json.dump(final_results, f, indent=4)

        print(f"====> HOÀN TẤT EPOCH {epoch}! Đã lưu kết quả tại: {output_json_path}")
        print(f"====> Tổng số UUID đã dự đoán: {len(final_results)}")

if __name__ == '__main__':
    infer_and_vote()