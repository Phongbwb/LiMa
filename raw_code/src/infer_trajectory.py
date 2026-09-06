import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import Counter, defaultdict # Thêm defaultdict để gom nhóm dự đoán

# Import chuẩn cho tính năng Mixed Precision (AMP)
from torch.cuda.amp import autocast

# Import Mô hình IntentPredictor
from models.classifier_trajectory import IntentPredictor

# ==========================================
# 1. BỘ DỮ LIỆU INFERENCE (LẤY 5 FRAME LIÊN TỤC ĐẾN HẾT)
# ==========================================
class VehicleIntentInferDataset(Dataset):
    def __init__(self, json_path, base_dir, transform=None):
        self.transform = transform
        self.base_dir = base_dir 
        self.data = [] # Lưu trữ từng CHUNK 5 frames độc lập
        self.seq_len = 5 

        print(f"====> Đang đọc dữ liệu Track Test từ: {json_path}...")
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
            for uuid_key, item in raw_data.items():
                frames = item["frames"]
                boxes = item["boxes"]
                
                assert len(frames) == len(boxes), f"Lỗi ở UUID {uuid_key}: Số frames khác boxes"
                
                # 🔥 LOGIC LẤY 5 FRAME CÁCH ĐỀU LIÊN TỤC TỪ SỐ FRAME CÒN LẠI
                available_indices = list(range(len(frames)))
                
                while len(available_indices) > 0:
                    if len(available_indices) >= self.seq_len:
                        # Lấy 5 index cách đều
                        idx_of_idx = np.linspace(0, len(available_indices) - 1, self.seq_len, dtype=int)
                        selected_indices = [available_indices[i] for i in idx_of_idx]
                        
                        # Xóa các index đã được chọn khỏi available_indices (xóa từ dưới lên để không lỗi index)
                        for i in sorted(idx_of_idx, reverse=True):
                            available_indices.pop(i)
                    else:
                        # Nếu số frame còn lại ít hơn 5, lấy tất cả và lặp lại frame cuối để đủ 5
                        selected_indices = available_indices.copy()
                        while len(selected_indices) < self.seq_len:
                            selected_indices.append(selected_indices[-1])
                        available_indices = [] # Kết thúc vòng lặp
                    
                    # Trích xuất data cho chunk này và thêm vào danh sách
                    chunk_frames = [frames[i] for i in selected_indices]
                    chunk_boxes = [boxes[i] for i in selected_indices]
                    
                    self.data.append({
                        "uuid": uuid_key,
                        "frames": chunk_frames,
                        "boxes": chunk_boxes
                    })

        print(f"====> Hoàn tất! Tổng số mẫu (chunks) cần dự đoán: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def _compute_trajectory_features(self, raw_boxes, img_w, img_h):
        """Tính toán 18 đặc trưng quỹ đạo"""
        T = len(raw_boxes)
        features = np.zeros((T, 18), dtype=np.float32)

        img_w = img_w if img_w > 0 else 1920
        img_h = img_h if img_h > 0 else 1080

        cx, cy, w, h = [], [], [], []
        for box in raw_boxes:
            cx.append((box[0] + box[2] / 2.0) / img_w) 
            cy.append((box[1] + box[3] / 2.0) / img_h)
            w.append(box[2] / img_w)
            h.append(box[3] / img_h)

        cx, cy, w, h = np.array(cx), np.array(cy), np.array(w), np.array(h)
        area = w * h
        aspect_ratio = w / (h + 1e-6)

        def diff(arr):
            d = np.zeros_like(arr)
            if len(arr) > 1:
                d[1:] = arr[1:] - arr[:-1]
                d[0] = d[1] 
            return d

        vx, vy = diff(cx), diff(cy)
        speed = np.sqrt(vx**2 + vy**2)

        sin_theta, cos_theta = np.zeros_like(speed), np.zeros_like(speed)
        mask = speed > 1e-6
        sin_theta[mask] = vy[mask] / speed[mask]
        cos_theta[mask] = vx[mask] / speed[mask]

        ax, ay = diff(vx), diff(vy)
        acc = np.sqrt(ax**2 + ay**2)
        jerk = diff(acc)

        sin_prev = np.roll(sin_theta, shift=1)
        cos_prev = np.roll(cos_theta, shift=1)
        if T > 1:
            sin_prev[0], cos_prev[0] = sin_prev[1], cos_prev[1]
            
        sin_diff = sin_theta * cos_prev - cos_theta * sin_prev
        cos_diff = cos_theta * cos_prev + sin_theta * sin_prev
        curvature = np.arctan2(sin_diff, cos_diff)

        area_change = diff(area)
        ratio_change = diff(aspect_ratio)

        features[:, 0], features[:, 1] = cx, cy
        features[:, 2], features[:, 3], features[:, 4] = vx, vy, speed
        features[:, 5], features[:, 6], features[:, 7] = ax, ay, acc
        features[:, 8], features[:, 9] = sin_theta, cos_theta
        features[:, 10], features[:, 11] = jerk, curvature
        features[:, 12], features[:, 13], features[:, 14], features[:, 15] = w, h, area, aspect_ratio
        features[:, 16], features[:, 17] = area_change, ratio_change

        return torch.tensor(features, dtype=torch.float32)

    def __getitem__(self, idx):
        # Do đã xử lý ở __init__, item giờ đây CHỈ LÀ 1 CHUNK 5 FRAMES
        item = self.data[idx]
        uuid_key = item["uuid"]
        chunk_frames = item["frames"]
        chunk_boxes = item["boxes"]

        crops = []
        img_w, img_h = 0, 0

        for path, box in zip(chunk_frames, chunk_boxes):
            full_path = os.path.join(self.base_dir, path)
            try:
                img = Image.open(full_path).convert("RGB")
                if img_w == 0:
                    img_w, img_h = img.size

                x, y, w, h = box
                x, y = max(0, int(x)), max(0, int(y))
                x2, y2 = min(img_w, int(x + w)), min(img_h, int(y + h))
                crop_img = img.crop((x, y, x2, y2))
                
                if self.transform:
                    crop_tensor = self.transform(crop_img)
                else:
                    crop_tensor = transforms.ToTensor()(crop_img)
            except Exception as e:
                print(f"Lỗi đọc ảnh {full_path}: {e}")
                crop_tensor = torch.zeros((3, 64, 64)) 
                
            crops.append(crop_tensor)

        # Shape: [C, T=5, H, W]
        crop_frames = torch.stack(crops, dim=1) 
        bbox_features = self._compute_trajectory_features(chunk_boxes, img_w, img_h)

        return crop_frames, bbox_features, uuid_key

# ==========================================
# 2. HÀM INFERENCE INTENT (HÀNH VI)
# ==========================================
def infer_intent():
    BASE_DIR = 'data/cityflownl/data' 
    TEST_JSON_PATH = './data/json/test-tracks.json' 
    WEIGHTS_PATH = './checkpoints/intent/intent_ep1.pth'
    OUTPUT_JSON_PATH = './data/json/test_tracks_intent.json'
    
    NUM_CLASSES = 3 
    EMBED_DIM = 256
    BATCH_SIZE = 32 # Có thể tăng Batch Size lên vì ta đã flatten các phần từ Dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n====> BẮT ĐẦU INFERENCE HÀNH VI (INTENT) TRÊN {device}")

    test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VehicleIntentInferDataset(TEST_JSON_PATH, base_dir=BASE_DIR, transform=test_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = IntentPredictor(d_model=EMBED_DIM, num_blocks=3).to(device)
    
    print(f"====> Đang tải weights từ: {WEIGHTS_PATH}")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    model.eval()

    # Lưu tất cả các dự đoán theo uuid
    uuid_predictions = defaultdict(list)

    print("\n====> Đang chạy dự đoán...")
    with torch.no_grad():
        for crop_frames, bbox_features, uuid_keys in tqdm(dataloader, desc="Dự đoán Batch"):
            # crop_frames shape lúc này là: [B, C, T_seq=5, H, W]
            crop_frames = crop_frames.to(device)
            bbox_features = bbox_features.to(device)
            
            with autocast():
                outputs = model(crop_frames, bbox_features)
                logits = outputs["logits"] # Shape: [B, NUM_CLASSES]
            
            _, preds = torch.max(logits, dim=1) # Shape: [B]
            preds = preds.cpu().numpy().tolist()
            
            # Đưa dự đoán vào dictionary theo uuid tương ứng
            for uuid, pred in zip(uuid_keys, preds):
                uuid_predictions[uuid].append(pred)

    print("\n====> Tiến hành Bầu Chọn (Majority Vote)...")
    final_results = {}
    for uuid, preds_list in uuid_predictions.items():
        vote_counts = Counter(preds_list)
        best_id = vote_counts.most_common(1)[0][0] # Lấy ID xuất hiện nhiều nhất
        final_results[uuid] = {"id": best_id}

    # ==========================================
    # 3. LƯU FILE JSON
    # ==========================================
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"\n====> HOÀN TẤT! Đã lưu kết quả tại: {OUTPUT_JSON_PATH}")
    print(f"Tổng số Track (UUID) đã dự đoán: {len(final_results)}\n")

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True 
        
    infer_intent()