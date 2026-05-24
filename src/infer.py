import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter

# Import ModelConfig và kiến trúc mô hình từ file train của bạn
from experiments.train_classifier import ModelConfig
from models.recognition_models import ResNet50_SingleTask

# ==========================================
# 1. BỘ DỮ LIỆU INFERENCE (ĐỌC TẤT CẢ FRAMES CỦA UUID)
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
                
                # Đảm bảo số lượng frame và box bằng nhau
                assert len(frames) == len(boxes), f"Lỗi ở UUID {uuid_key}: Số lượng frames khác boxes"
                
                # Trải phẳng dữ liệu: Mỗi frame bây giờ là 1 sample, nhưng giữ lại thông tin UUID
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
        
        try:
            full_image = Image.open(item["full_path"]).convert("RGB")
            # Cắt ảnh theo box của frame đó
            x, y, w, h = item["box"]
            crop_image = full_image.crop((x, y, x + w, y + h)) 
        except Exception as e:
            # Nếu lỗi, tạo ảnh đen
            crop_image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Áp dụng Transform
        if self.transform:
            crop_tensor = self.transform(crop_image)
        else:
            crop_tensor = transforms.ToTensor()(crop_image)

        return crop_tensor, uuid_key

# ==========================================
# 2. HÀM INFERENCE CÓ CƠ CHẾ VOTING
# ==========================================
def infer_and_vote():
    # ---------------------------------------------------------
    # CẤU HÌNH ĐƯỜNG DẪN 
    # ---------------------------------------------------------
    TARGET_TASK = 'direction' # 'color', 'type', hoặc 'direction'
    BASE_DIR = 'data/cityflownl/data' 
   
    # File JSON chứa dữ liệu test (Cấu trúc giống file train)     
    TEST_JSON_PATH = './data/json/test-tracks.json' 

    # File weights (.pth) bạn đã train xong
    WEIGHTS_PATH = f'./checkpoints/recognition/single_{TARGET_TASK}_resnet50_ep4.pth' 

    # File JSON kết quả đầu ra
    OUTPUT_JSON_PATH = f'./data/json/test_queries_{TARGET_TASK}.json'
    # ---------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n====> BẮT ĐẦU INFERENCE TASK: {TARGET_TASK.upper()} TRÊN {device}")

    # 1. Pipeline tiền xử lý (Không có Augmentation)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    # 2. Tải Dataset và DataLoader
    dataset = VehicleTrackDataset(TEST_JSON_PATH, base_dir=BASE_DIR, transform=test_transforms)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Khởi tạo Model
    cfg = ModelConfig(task=TARGET_TASK)
    model = ResNet50_SingleTask(cfg).to(device)
    
    print(f"====> Đang tải weights từ: {WEIGHTS_PATH}")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    # Dictionary để lưu danh sách tất cả các dự đoán của từng UUID
    # Ví dụ: uuid_predictions = {"uuid1": [0, 0, 0, 1, 0], "uuid2": [2, 2, 2]}
    uuid_predictions = {}

    print("====> Đang chạy mô hình dự đoán từng frame...")
    with torch.no_grad():
        for i, (crops, uuid_keys) in enumerate(dataloader):
            crops = crops.to(device)
            logits = model(crops)
            
            _, preds = torch.max(logits, dim=1)
            preds = preds.cpu().numpy().tolist()
            
            # Ghi nhận kết quả dự đoán vào danh sách của UUID tương ứng
            for uuid, pred_id in zip(uuid_keys, preds):
                if uuid not in uuid_predictions:
                    uuid_predictions[uuid] = []
                uuid_predictions[uuid].append(pred_id)
                
            if (i + 1) % 10 == 0:
                print(f"  + Đã xử lý Batch [{i+1}/{len(dataloader)}]")

    # ==========================================
    # 4. MAJORITY VOTING (TÌM KẾT QUẢ XUẤT HIỆN NHIỀU NHẤT)
    # ==========================================
    print("\n====> Tiến hành Majority Voting cho từng UUID...")
    final_results = {}
    
    for uuid, list_preds in uuid_predictions.items():
        # Dùng thư viện Counter để đếm tần suất xuất hiện của các ID
        vote_counts = Counter(list_preds)
        
        # Lấy ID có số phiếu cao nhất (phần tử đầu tiên của most_common)
        best_id = vote_counts.most_common(1)[0][0]
        
        final_results[uuid] = {"id": best_id}

    # ==========================================
    # 5. LƯU FILE JSON
    # ==========================================
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"====> HOÀN TẤT! Đã lưu kết quả tại: {OUTPUT_JSON_PATH}")
    print(f"Tổng số UUID đã dự đoán: {len(final_results)}")

if __name__ == '__main__':
    infer_and_vote()