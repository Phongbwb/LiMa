import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Import mô hình từ file bên ngoài (giống code gốc của bạn)
from models.recognition_models import ResNet50_SingleTask

# ==========================================
# 1. CẤU HÌNH (CONFIG)
# ==========================================
class ModelConfig:
    def __init__(self, task='color'):
        self.TASK = task.lower() # 'color' hoặc 'type'
        
        if self.TASK == 'color':
            self.NUM_CLASSES = 8
            self.JSON_PATH = './data/json/color_trainset_train.json'
        elif self.TASK == 'type':
            self.NUM_CLASSES = 7
            self.JSON_PATH = './data/json/type_trainset_train.json'
        
        elif self.TASK == 'direction':
            self.NUM_CLASSES = 4   # 0: go straight, 1: turn right, 2: turn left, 3: stop
            self.JSON_PATH = './data/json/direction_trainset_train.json' # Thay đổi đường dẫn tới file JSON của bạn
            
        else:
            raise ValueError("Task chỉ có thể là: 'color', 'type', hoặc 'direction'")
            
        self.EMBED_DIM = 512
        self.RESNET_CHECKPOINT = "" 

# ==========================================
# 2. BỘ DỮ LIỆU (DATASET)
# ==========================================
class VehicleSingleTaskDataset(Dataset):
    def __init__(self, json_path, base_dir, transform=None):
        self.transform = transform
        self.base_dir = base_dir 
        self.data = []

        print(f"====> Đang đọc dữ liệu từ: {json_path}...")
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
            for key, item in raw_data.items():
                img_path = os.path.join(self.base_dir, item["frames"])
                
                # Check file tồn tại để tránh lỗi
                if os.path.exists(img_path):
                    self.data.append({
                        "full_path": img_path,
                        "boxes": item["boxes"],
                        "label": item["id"]
                    })

        print(f"====> Hoàn tất! Số lượng mẫu huấn luyện hợp lệ: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_image = Image.open(item["full_path"]).convert("RGB")
            
        x, y, w, h = item["boxes"]
        crop_image = full_image.crop((x, y, x + w, y + h)) 
        
        label = torch.tensor(item["label"], dtype=torch.long)

        if self.transform:
            crop_tensor = self.transform(crop_image)
        else:
            crop_tensor = transforms.ToTensor()(crop_image)

        return crop_tensor, label

# ==========================================
# 3. VÒNG LẶP HUẤN LUYỆN
# ==========================================
def train():
    # ---------------------------------------------------------
    # CHỌN TÁC VỤ TẠI ĐÂY: 'color' HOẶC 'type'
    TARGET_TASK = 'direction'  # Thay bằng 'color' hoặc 'type' tùy bạn muốn train tác vụ nào
    # ---------------------------------------------------------
    
    BASE_DIR = 'data/cityflownl/data' 
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = ModelConfig(task=TARGET_TASK)
    print(f"\n[{'*'*30}]")
    print(f" THIẾT BỊ HUẤN LUYỆN : {device}")
    print(f" TÁC VỤ              : {cfg.TASK.upper()}")
    print(f"[{'*'*30}]\n")

    train_transforms = transforms.Compose([
        transforms.Resize((336, 336)),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    dataset = VehicleSingleTaskDataset(cfg.JSON_PATH, base_dir=BASE_DIR, transform=train_transforms)
    
    if len(dataset) == 0:
        print("LỖI: Không tìm thấy dữ liệu. Kiểm tra lại BASE_DIR và JSON_PATH.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # KHỞI TẠO MÔ HÌNH (Gọi từ file bên ngoài)
    model = ResNet50_SingleTask(cfg).to(device)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    os.makedirs('./checkpoints/recognition/', exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (crops, labels) in enumerate(dataloader):
            crops = crops.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(crops)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{i+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = (correct / total) * 100
        
        print(f"\n---> EPOCH {epoch+1} HOÀN TẤT | LOSS: {epoch_loss:.4f} | ACCURACY: {epoch_acc:.2f}%")

        checkpoint_path = f'./checkpoints/recognition/single_{cfg.TASK}_resnet50_ep{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"---> Đã lưu: {checkpoint_path}\n")

if __name__ == '__main__':
    train()