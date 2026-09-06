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
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from models.classifier_mamba import PureColorMambaClassifier

# ==========================================
# 1. CONFIG
# ==========================================
class ModelConfig:
    def __init__(self):
        self.TASK = 'color'
        self.NUM_CLASSES = 8       # Số lượng nhãn màu
        self.EMBED_DIM = 256       # Chiều vector đặc trưng
        self.NUM_MAMBA_BLOCKS = 6  # Số khối Dense-Mamba
        
        self.BASE_DIR = 'data/cityflownl/data'
        self.TRAIN_JSON = './data/json/color_trainset_train.json'
        self.VAL_RATIO = 0.2       # Trích xuất 20% dữ liệu làm tập Validation
        
        self.BATCH_SIZE = 32
        self.EPOCHS = 30
        self.MAX_LR = 1e-3         # Đỉnh của OneCycleLR
        self.WEIGHT_DECAY = 1e-3
        self.NUM_WORKERS = 4
        self.SAVE_DIR = './checkpoints/recognition/'

# ==========================================
# 2. HÀM CHIA DỮ LIỆU TỰ ĐỘNG (ON-THE-FLY SPLIT)
# ==========================================
def get_train_val_splits(json_path, base_dir, val_ratio=0.2, seed=42):
    print(f"====> Đang đọc dữ liệu từ: {json_path}...")
    try:
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f" Lỗi: Không tìm thấy file {json_path}")
        return [], []

    valid_data = []
    for key, item in raw_data.items():
        img_path = os.path.join(base_dir, item["frames"])
        if os.path.exists(img_path):
            valid_data.append({
                "full_path": img_path,
                "boxes": item["boxes"],
                "label": item["id"]
            })

    print(f"====> Đã tải {len(valid_data)} mẫu hợp lệ.")

    # Xáo trộn dữ liệu với Seed cố định
    random.seed(seed)
    random.shuffle(valid_data)
    
    # Chia tỷ lệ
    split_idx = int(len(valid_data) * (1 - val_ratio))
    train_list = valid_data[:split_idx]
    val_list = valid_data[split_idx:]
    
    return train_list, val_list

# ==========================================
# 3. FOCAL LOSS DÀNH CHO LONG-TAIL 
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, device='cuda'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device) if alpha is not None else None

    def forward(self, inputs, targets):
        # Tính CE thuần để lấy xác suất pt chính xác
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Công thức Focal Loss cốt lõi
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Nhân trọng số lớp alpha (nếu có)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

# ==========================================
# 4. BỘ DỮ LIỆU (DATASET)
# ==========================================
class PureColorDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.transform = transform
        self.data = data_list
        self.labels_list = [item["label"] for item in self.data]

    def __len__(self):
        return len(self.data)

    def get_sampler_weights(self, num_classes):
        # Đã cập nhật công thức chuẩn (Scikit-Learn) để không làm nổ Loss
        class_counts = np.bincount(self.labels_list, minlength=num_classes)
        class_counts_safe = np.maximum(class_counts, 1) # Chống chia cho 0
        total_samples = len(self.labels_list)
        
        # Công thức: total / (num_classes * count)
        class_weights = total_samples / (num_classes * class_counts_safe)
        
        sample_weights = [class_weights[label] for label in self.labels_list]
        return torch.DoubleTensor(sample_weights), class_weights

    def __getitem__(self, idx):
        item = self.data[idx]
        full_image = Image.open(item["full_path"]).convert("RGB")
            
        x, y, w, h = item["boxes"]
        # Padding nhẹ 5% để lấy bối cảnh viền xe
        padding_x, padding_y = int(w * 0.05), int(h * 0.05)
        crop_image = full_image.crop((
            max(0, x - padding_x), 
            max(0, y - padding_y), 
            min(full_image.width, x + w + padding_x), 
            min(full_image.height, y + h + padding_y)
        )) 
        
        label = torch.tensor(item["label"], dtype=torch.long)

        if self.transform:
            crop_tensor = self.transform(crop_image)
        else:
            crop_tensor = transforms.ToTensor()(crop_image)
            
        return crop_tensor, label

# ==========================================
# 5. HÀM ĐÁNH GIÁ (VALIDATION)
# ==========================================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Khởi chạy Autocast (bỏ tham số 'cuda')
            with autocast():
                logits = model(inputs)
                loss = criterion(logits, labels)
            
            val_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return val_loss / len(dataloader), (correct / total) * 100

# ==========================================
# 6. VÒNG LẶP HUẤN LUYỆN CHÍNH
# ==========================================
def train():
    cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    print(f"\n[{'='*40}]")
    print(f" THIẾT BỊ     : {device}")
    print(f" BATCH SIZE   : {cfg.BATCH_SIZE}")
    print(f" LỚP MÀU SẮC  : {cfg.NUM_CLASSES}")
    print(f"[{'='*40}]\n")

    # --- CHIA DỮ LIỆU ĐỘNG ---
    train_data_list, val_data_list = get_train_val_splits(
        json_path=cfg.TRAIN_JSON, 
        base_dir=cfg.BASE_DIR, 
        val_ratio=cfg.VAL_RATIO,
        seed=42
    )
    
    if len(train_data_list) == 0:
        return

    print(f"Tổng số ảnh phân bổ  : {len(train_data_list) + len(val_data_list)}")
    print(f" ├── Tập Huấn luyện  : {len(train_data_list)} ảnh")
    print(f" └── Tập Đánh giá    : {len(val_data_list)} ảnh\n")

    # --- AUGMENTATION ---
    train_transforms = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.CenterCrop((336, 336)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.015),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.CenterCrop((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- DATALOADER ---
    train_dataset = PureColorDataset(train_data_list, train_transforms)
    val_dataset = PureColorDataset(val_data_list, val_transforms)

    sample_weights, class_weights_raw = train_dataset.get_sampler_weights(cfg.NUM_CLASSES)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, sampler=sampler, 
                              num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, 
                            num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # --- KHỞI TẠO MÔ HÌNH ---
    model = PureColorMambaClassifier(
        num_colors=cfg.NUM_CLASSES, 
        d_model=cfg.EMBED_DIM
    ).to(device)

    # Truyền trọng số tĩnh trực tiếp (Không chia np.max nữa)
    alpha_weights = class_weights_raw 
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0, device=device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.MAX_LR, weight_decay=cfg.WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=cfg.MAX_LR, 
        steps_per_epoch=len(train_loader), 
        epochs=cfg.EPOCHS
    )
    
    scaler = GradScaler() 

    best_val_acc = 0.0

    # --- VÒNG LẶP HUẤN LUYỆN ---
    for epoch in range(cfg.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Khởi chạy Autocast (bỏ tham số 'cuda')
            with autocast():
                logits = model(inputs)
                loss = criterion(logits, labels)

            # Backward pass
            scaler.scale(loss).backward()
            
            # Clip gradient
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Tính toán Metric
            running_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}", 
                'Acc': f"{(correct/total)*100:.1f}%",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })

        train_loss = running_loss / len(train_loader)
        train_acc = (correct / total) * 100
        
        # Đánh giá trên tập Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"\n Kết quả Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")

        # --- LƯU TRỌNG SỐ CHO TỪNG EPOCH ---
        epoch_path = os.path.join(cfg.SAVE_DIR, f'mamba_{cfg.TASK}_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), epoch_path)
        print(f" Đã lưu checkpoint riêng cho Epoch {epoch+1} tại: {epoch_path}")
        
        # Lưu mô hình tốt nhất (Best Model)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(cfg.SAVE_DIR, f'best_model_ep{epoch+1}_{val_acc:.1f}.pth')
            torch.save(model.state_dict(), best_path)
            print(f" Tìm thấy mô hình tốt nhất mới! Đã lưu tại: {best_path}\n")
        else:
            print("") # Xuống dòng cho đẹp log

    print(f" Hoàn tất huấn luyện! Độ chính xác tập Val cao nhất thu được: {best_val_acc:.2f}%")

if __name__ == '__main__':
    # Cố định Seed để kết quả có thể tái lập
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True 
        
    train()