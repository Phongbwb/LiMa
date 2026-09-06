import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.cuda.amp import GradScaler, autocast

# 🔥 Import mô hình từ file của bạn (Đảm bảo đường dẫn import chính xác)
from models.classifier_trajectory import IntentPredictor 

# ==========================================
# 1. CẤU HÌNH (CONFIG)
# ==========================================
class ModelConfig:
    def __init__(self):
        self.TASK = 'intent'
        self.NUM_CLASSES = 3      # Chỉ lấy 0 (Thẳng), 1 (Phải), 2 (Trái)
        self.TRAIN_JSON = './data/json/intent_trainset_5frames_sliding.json' 
        self.BASE_DIR = 'data/cityflownl/data'
        self.SEQ_LEN = 5          
        self.D_MODEL = 256
        self.CROP_SIZE = (128, 128)
        self.BATCH_SIZE = 64
        self.EPOCHS = 10
        self.LR = 1e-4

# ==========================================
# 2. HÀM ĐỒNG BỘ AUGMENTATION (SYNC AUGMENT)
# ==========================================
def augment_multimodal_sequence(images, boxes, label, img_w, img_h, flip_prob=0.1, jitter_prob=0.1, jitter_limit=3):
    aug_images = [img.copy() for img in images]
    aug_boxes = [list(box) for box in boxes]
    aug_label = label

    # 1. LẬT NGANG CÓ KIỂM SOÁT
    if random.random() < flip_prob:
        aug_images = [TF.hflip(img) for img in aug_images]
        for i in range(len(aug_boxes)):
            x, y, w, h = aug_boxes[i]
            new_x = img_w - (x + w)
            aug_boxes[i] = [new_x, y, w, h]
        # Đảo nhãn vật lý
        if aug_label == 1: aug_label = 2
        elif aug_label == 2: aug_label = 1

    # 2. KINEMATIC JITTERING
    if random.random() < jitter_prob:
        for i in range(len(aug_boxes)):
            x, y, w, h = aug_boxes[i]
            noise_x = random.uniform(-jitter_limit, jitter_limit)
            noise_y = random.uniform(-jitter_limit, jitter_limit)
            noise_w = random.uniform(-jitter_limit, jitter_limit)
            noise_h = random.uniform(-jitter_limit, jitter_limit)
            
            new_x = max(0, min(img_w - 1, x + noise_x))
            new_y = max(0, min(img_h - 1, y + noise_y))
            new_w = max(1, min(img_w - new_x, w + noise_w))
            new_h = max(1, min(img_h - new_y, h + noise_h))
            aug_boxes[i] = [new_x, new_y, new_w, new_h]

    return aug_images, aug_boxes, aug_label

# ==========================================
# 3. BỘ DỮ LIỆU & TÍNH ĐẶC TRƯNG QUỸ ĐẠO
# ==========================================
class IntentVideoDataset(Dataset):
    def __init__(self, json_path, base_dir, transform=None):
        self.transform = transform
        self.base_dir = base_dir 
        self.data = []
        self.labels_list = [] 

        print(f"====> Đang load 100% Data từ: {json_path}")
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            for key, item in raw_data.items():
                label_id = item["label"]
                if label_id not in [0, 1, 2]: # Lọc bỏ nhãn 3 (Dừng/Khác)
                    continue 

                self.data.append({
                    "frames": item["frames"],
                    "boxes": item["boxes"],   
                    "label": label_id       
                })
                self.labels_list.append(label_id)

    def __len__(self):
        return len(self.data)

    def get_sampler_weights(self):
        # Đếm số lượng mẫu thực tế của mỗi class trong dataset
        class_counts = np.bincount(self.labels_list, minlength=3)
        
        # 🔥 Xác suất bốc trúng mục tiêu: Thẳng 20%, Phải 40%, Trái 40%
        target_probs = np.array([0.2, 0.4, 0.4])
        
        # Tính trọng số cho từng class (Cộng 1e-5 để tránh lỗi chia cho 0 nếu class trống)
        class_weights = target_probs / (class_counts + 1e-5)
        
        # Gán trọng số tương ứng cho từng sample trong dataset
        sample_weights = [class_weights[label] for label in self.labels_list]
        
        return torch.DoubleTensor(sample_weights)

    def _compute_trajectory_features(self, raw_boxes, img_w, img_h):
        T = len(raw_boxes)
        features = np.zeros((T, 18), dtype=np.float32)

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

        sin_prev, cos_prev = np.roll(sin_theta, shift=1), np.roll(cos_theta, shift=1)
        if T > 1: sin_prev[0], cos_prev[0] = sin_prev[1], cos_prev[1]
            
        sin_diff = sin_theta * cos_prev - cos_theta * sin_prev
        cos_diff = cos_theta * cos_prev + sin_theta * sin_prev
        curvature = np.arctan2(sin_diff, cos_diff)

        area_change, ratio_change = diff(area), diff(aspect_ratio)

        features[:, 0:2] = np.column_stack((cx, cy))
        features[:, 2:5] = np.column_stack((vx, vy, speed))
        features[:, 5:8] = np.column_stack((ax, ay, acc))
        features[:, 8:10] = np.column_stack((sin_theta, cos_theta))
        features[:, 10:12] = np.column_stack((jerk, curvature))
        features[:, 12:16] = np.column_stack((w, h, area, aspect_ratio))
        features[:, 16:18] = np.column_stack((area_change, ratio_change))

        return torch.tensor(features, dtype=torch.float32)

    def __getitem__(self, idx):
        item = self.data[idx]
        frame_paths = item["frames"]
        raw_boxes = item["boxes"]
        label = item["label"]
        
        raw_images_sequence = []
        img_w, img_h = 0, 0
        
        # 1. Đọc ảnh gốc
        for path in frame_paths:
            full_path = os.path.join(self.base_dir, path)
            try:
                img = Image.open(full_path).convert("RGB")
            except:
                img = Image.new("RGB", (1920, 1080), (128, 128, 128)) 
            if img_w == 0:
                img_w, img_h = img.size
            raw_images_sequence.append(img)

        # 2. Gọi Sync Augmentation cho mẫu Rẽ (1, 2)
        if label in [1, 2] and self.transform is not None:
            aug_images, final_boxes, final_label = augment_multimodal_sequence(
                raw_images_sequence, raw_boxes, label, img_w, img_h
            )
        else:
            aug_images, final_boxes, final_label = raw_images_sequence, raw_boxes, label

        # 3. Crop & Transform
        crops = []
        for img, box in zip(aug_images, final_boxes):
            x, y, w, h = box
            x, y = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img_w, int(x + w)), min(img_h, int(y + h))
            
            crop_img = img.crop((x, y, x2, y2))
            crop_tensor = self.transform(crop_img) if self.transform else transforms.ToTensor()(crop_img)
            crops.append(crop_tensor)

        crop_frames = torch.stack(crops, dim=1) 
        bbox_features = self._compute_trajectory_features(final_boxes, img_w, img_h) 
        label_tensor = torch.tensor(final_label, dtype=torch.long)
        
        return crop_frames, bbox_features, label_tensor

# ==========================================
# 4. FOCAL LOSS
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma 
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss

# ==========================================
# 5. TRAINING LOOP (FULL DATA, SAVE EVERY EPOCH)
# ==========================================
def train():
    cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chuẩn hóa ảnh (Color Jitter sẽ chạy độc lập, Flip/Box Jitter đã xử lý ở trên)
    train_transforms = transforms.Compose([
        transforms.Resize(cfg.CROP_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    # Khởi tạo Dataset 100%
    train_dataset = IntentVideoDataset(cfg.TRAIN_JSON, cfg.BASE_DIR, transform=train_transforms)

    # Cân bằng dữ liệu với WeightedRandomSampler
    sample_weights = train_dataset.get_sampler_weights()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights)//20, replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)

    # Model, Loss, Optimizer, Scaler
    model = IntentPredictor(d_model=cfg.D_MODEL, num_blocks=3).to(device)
    criterion = FocalLoss(gamma=2.0) 
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    scaler = GradScaler() 

    os.makedirs('./checkpoints/intent/', exist_ok=True)

    print(f"\n[{'*'*40}]")
    print(f" BẮT ĐẦU HUẤN LUYỆN (FULL DATA) TRÊN {device}")
    print(f" TỔNG EPOCH: {cfg.EPOCHS} | BATCH: {cfg.BATCH_SIZE}")
    print(f" TRAIN SAMPLES: {len(train_dataset)}")
    print(f"[{'*'*40}]\n")

    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for i, (crop_frames, bbox_features, labels) in enumerate(train_loader):
            crop_frames, bbox_features, labels = crop_frames.to(device), bbox_features.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(crop_frames, bbox_features)
                logits = outputs['logits']
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            
            # Tính accuracy cơ bản trên mini-batch
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{cfg.EPOCHS}] Batch [{i+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # Cập nhật Learning Rate
        scheduler.step()
        
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = (correct / total) * 100
        print(f"---> KẾT THÚC EPOCH {epoch+1} | Mean Loss: {epoch_loss:.4f} | Train Acc (Mixed): {epoch_acc:.2f}%")

        # ------------------- SAVE CHECKPOINT -------------------
        # Lưu checkpoint sau MỖI epoch
        checkpoint_path = f'./checkpoints/intent/intent_ep{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Đã lưu mốc: {checkpoint_path}")
            
    # Lưu Model cuối cùng sau khi train xong
    final_path = './checkpoints/intent/intent_final.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\n🎉 HOÀN TẤT HUẤN LUYỆN! Đã lưu model cuối tại: {final_path}")

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True 
        
    train()