import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# Import model từ file intent_predictor_v2.py
# (đặt 2 file cùng thư mục hoặc chỉnh sys.path cho phù hợp)
from models.classifier_type import IntentPredictor

# ==========================================
# 1. CONFIG — CỐ ĐỊNH task = 'type'
# ==========================================
class TypeConfig:
    TASK        = 'type'
    NUM_CLASSES = 7           # 7 loại xe trong dataset
    JSON_PATH   = './data/json/type_trainset_train.json'

    # Model
    D_MODEL     = 256
    NUM_BLOCKS  = 3           # số SpatialEncoderLayer

    # Train
    BATCH_SIZE  = 32
    EPOCHS      = 30
    LR          = 3e-5        # backbone: 3e-6, head: 3e-5 (xem optimizer bên dưới)
    WEIGHT_DECAY= 1e-4
    IMG_SIZE    = 224


# ==========================================
# 2. DATASET
# ==========================================
class VehicleTypeDataset(Dataset):
    """
    Dataset chỉ cho task phân loại LOẠI XE (type).
    Mỗi mẫu trả về: crop_tensor, label
    Giữ nguyên toàn bộ class kể cả class rất ít mẫu.
    Dùng sqrt-smoothing + cap để hệ số lấy mẫu không quá chênh lệch.
    """
    def __init__(self, cfg, base_dir, transform=None):
        self.cfg             = cfg
        self.transform       = transform
        self.base_dir        = base_dir
        self.data            = []
        self.labels_list     = []
        self.num_classes_actual = cfg.NUM_CLASSES

        print(f"====> Đang đọc dữ liệu từ: {cfg.JSON_PATH} ...")
        with open(cfg.JSON_PATH, 'r') as f:
            raw_data = json.load(f)

        for key, item in raw_data.items():
            img_path = os.path.join(base_dir, item["frames"])
            if not os.path.exists(img_path):
                continue
            self.data.append({
                "full_path": img_path,
                "boxes":     item["boxes"],
                "label":     item["id"],
            })
            self.labels_list.append(item["id"])

        counts = np.bincount(self.labels_list, minlength=cfg.NUM_CLASSES)
        print(f"====> Hoàn tất! Số mẫu hợp lệ: {len(self.data)}")
        print(f"      Class ít nhất: {counts.min()} mẫu | "
              f"Class nhiều nhất: {counts.max()} mẫu")

    def __len__(self):
        return len(self.data)

    def get_sampler_weights(self):
        """
        Trọng số WeightedRandomSampler với sqrt-smoothing + cap.

        Thay vì w = 1/n (làm class 1 mẫu được bốc gấp 39751 lần class lớn nhất),
        dùng w = 1/sqrt(n) để nén tỉ lệ chênh lệch xuống,
        sau đó cap tại MAX_RATIO lần class trung vị.

        Ví dụ với dữ liệu thực tế:
          Class 1 (n=39751): w_raw = 1/sqrt(39751) = 0.005
          Class 5 (n=1):     w_raw = 1/sqrt(1)     = 1.0   → tỉ lệ = 200x
          Sau cap MAX_RATIO=20: class 5 được bốc tối đa 20x class trung vị
        """
        MAX_RATIO = 2.0    # class hiếm nhất được bốc tối đa 2x class trung vị
                           # Cap thấp để tránh overfit class chỉ có 1-2 mẫu

        counts = np.bincount(self.labels_list, minlength=self.cfg.NUM_CLASSES)
        counts = np.where(counts == 0, 1, counts)          # tránh chia 0

        # sqrt-smoothing: nén khoảng cách giữa class lớn và class nhỏ
        w_per_class = 1.0 / np.sqrt(counts.astype(float))

        # Cap: tính ngưỡng từ class trung vị, giới hạn class hiếm
        median_w = np.median(w_per_class[counts > 1])      # bỏ qua class 0 mẫu khi tính median
        cap_val  = median_w * MAX_RATIO
        w_per_class = np.clip(w_per_class, 0, cap_val)

        # In bảng để dễ kiểm tra
        print("\n====> Sampler weight per class (sqrt + cap):")
        for c in range(self.cfg.NUM_CLASSES):
            print(f"    Class {c}: n={counts[c]:6d}  w={w_per_class[c]:.6f}")

        sample_weights = np.array([w_per_class[l] for l in self.labels_list])
        return torch.DoubleTensor(sample_weights)

    def get_class_weights(self):
        """
        Trọng số cho CrossEntropyLoss(weight=...) với sqrt-smoothing + cap.
        Cùng logic với sampler để 2 lớp nhất quán với nhau.
        Cap nhỏ hơn sampler (MAX_WEIGHT=5) vì loss weight ảnh hưởng trực tiếp
        đến gradient — cần thận trọng hơn.
        """
        MAX_WEIGHT = 2.0   # cap thấp để class 1 mẫu không khuếch đại gradient quá mức

        counts  = np.bincount(self.labels_list, minlength=self.cfg.NUM_CLASSES)
        counts  = np.where(counts == 0, 1, counts)
        weights = 1.0 / np.sqrt(counts.astype(float))
        # Normalize theo median (nhất quán với sampler), rồi mới cap
        median_w = np.median(weights[counts > 1])
        weights  = weights / median_w              # median class = 1.0
        weights  = np.clip(weights, 0, MAX_WEIGHT)

        print(f"\n====> Class weight cho loss (sqrt + cap={MAX_WEIGHT}):")
        print(f"  {'Class':<8} {'Số mẫu':>8}  {'Phân phối':<22}  {'Weight':>8}")
        print(f"  {'-'*54}")
        total_max = max(np.bincount(self.labels_list, minlength=self.cfg.NUM_CLASSES))
        for c in range(self.cfg.NUM_CLASSES):
            n   = int(np.bincount(self.labels_list, minlength=self.cfg.NUM_CLASSES)[c])
            bar = '█' * int(n / total_max * 20)
            print(f"  Class {c}:  {n:5d}  {bar:<22s}  {weights[c]:>7.4f}")
        print()

        return torch.FloatTensor(weights)

    def __getitem__(self, idx):
        item       = self.data[idx]
        full_image = Image.open(item["full_path"]).convert("RGB")

        x, y, w, h = item["boxes"]
        crop_image  = full_image.crop((x, y, x + w, y + h))

        if self.transform:
            crop_tensor = self.transform(crop_image)
        else:
            crop_tensor = transforms.ToTensor()(crop_image)

        label = torch.tensor(item["label"], dtype=torch.long)
        return crop_tensor, label


# ==========================================
# 3. AUGMENTATION CHUYÊN BIỆT CHO LOẠI XE
#
#  Mục tiêu: phân biệt xe máy / ô tô / xe tải / xe buýt ...
#  → Giữ nguyên tỉ lệ khung hình, màu sắc, góc nhìn
#  → KHÔNG flip ngang (lật xe sẽ gây nhầm hướng đầu/đuôi xe)
#  → Augmentation nhẹ để tránh xóa mất đặc trưng hình dạng
# ==========================================
def build_transform(cfg, is_train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )

    if is_train:
        return transforms.Compose([
            # Resize giữ tỉ lệ, sau đó center-crop
            transforms.Resize(int(cfg.IMG_SIZE * 1.12)),
            transforms.CenterCrop(cfg.IMG_SIZE),

            # --- Augmentation hình dạng ---
            # Task TYPE: phân biệt loại xe (ô tô / xe máy / xe tải...)
            # → hình dạng tổng thể quan trọng, không cần phân biệt đầu/đuôi
            # → lật ngang thoải mái, giúp tăng gấp đôi đa dạng dữ liệu
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),          # nghiêng ±10° (tăng lên từ ±5°)
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4),

            # --- Augmentation màu sắc ---
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05
            ),

            transforms.ToTensor(),
            normalize,

            # Che ngẫu nhiên — sau ToTensor
            transforms.RandomErasing(
                p=0.3,
                scale=(0.02, 0.10),
                ratio=(0.3, 3.3),
                value='random'
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(cfg.IMG_SIZE * 1.12)),
            transforms.CenterCrop(cfg.IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ])


# ==========================================
# 4. VÒNG LẶP TRAIN 1 EPOCH
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs, num_classes):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    # Per-class tracking để theo dõi class hiếm
    class_correct = torch.zeros(num_classes)
    class_total   = torch.zeros(num_classes)

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        out  = model(images)
        loss = criterion(out["logits"], labels)

        loss.backward()
        # Gradient clipping: tránh gradient explode khi loss lớn đột ngột
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        preds    = out["logits"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        # Tích lũy per-class accuracy
        for c in range(num_classes):
            mask = (labels == c)
            class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            class_total[c]   += mask.sum().item()

        if (i + 1) % 20 == 0:
            print(f"  [Epoch {epoch+1}/{total_epochs}] "
                  f"Batch {i+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {correct/total*100:.2f}%")

    # In per-class accuracy sau mỗi epoch
    print(f"\n  Per-class accuracy (epoch {epoch+1}):")
    for c in range(num_classes):
        n   = int(class_total[c].item())
        acc = class_correct[c].item() / n * 100 if n > 0 else 0.0
        bar = '█' * int(acc / 100 * 20)
        print(f"    Class {c}: {acc:5.1f}%  {bar:<20s}  ({n} mẫu)")

    return running_loss / len(loader), correct / total * 100


# ==========================================
# 5. MAIN TRAIN
# ==========================================
def train():
    cfg    = TypeConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = 'data/cityflownl/data'

    print(f"\n{'='*40}")
    print(f"  TASK        : {cfg.TASK.upper()} ({cfg.NUM_CLASSES} classes)")
    print(f"  DEVICE      : {device}")
    print(f"  BATCH SIZE  : {cfg.BATCH_SIZE}")
    print(f"  EPOCHS      : {cfg.EPOCHS}")
    print(f"  IMG SIZE    : {cfg.IMG_SIZE}")
    print(f"{'='*40}\n")

    # --- Dataset & DataLoader ---
    train_transform = build_transform(cfg, is_train=True)
    dataset = VehicleTypeDataset(cfg, base_dir=BASE_DIR, transform=train_transform)

    if len(dataset) == 0:
        print(" Không tìm thấy dữ liệu hợp lệ. Kiểm tra lại JSON_PATH và BASE_DIR.")
        return

    # WeightedRandomSampler với sqrt-smoothing + cap
    sampler = WeightedRandomSampler(
        weights     = dataset.get_sampler_weights(),
        num_samples = len(dataset),
        replacement = True
    )

    loader = DataLoader(
        dataset,
        batch_size  = cfg.BATCH_SIZE,
        sampler     = sampler,
        num_workers = 4,
        pin_memory  = True,
    )

    # --- Model ---
    model = IntentPredictor(
        d_model     = cfg.D_MODEL,
        num_blocks  = cfg.NUM_BLOCKS,
        num_classes = cfg.NUM_CLASSES,
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Tổng params      : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}\n")

    # --- Loss ---
    class_weights = dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimizer: 2 group LR ---
    backbone_params = list(model.backbone.parameters())
    head_params     = (
        list(model.spatial_encoder.parameters()) +
        list(model.gem_pool.parameters()) +
        list(model.classifier.parameters())
    )
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": cfg.LR * 0.1},
        {"params": head_params,     "lr": cfg.LR},
    ], weight_decay=cfg.WEIGHT_DECAY)

    # --- Checkpoint dir ---
    ckpt_dir = './checkpoints/type/'
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Training loop ---
    for epoch in range(cfg.EPOCHS):
        epoch_loss, epoch_acc = train_one_epoch(
            model, loader, criterion, optimizer, device,
            epoch, cfg.EPOCHS, cfg.NUM_CLASSES
        )

        print(f"\n---> EPOCH {epoch+1}/{cfg.EPOCHS} | "
              f"LOSS: {epoch_loss:.4f} | "
              f"ACC: {epoch_acc:.2f}%")

        ckpt_path = os.path.join(
            ckpt_dir, f"type_efficientnet_mamba_ep{epoch+1:02d}.pth"
        )
        torch.save({
            "epoch":       epoch + 1,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "loss":        epoch_loss,
            "acc":         epoch_acc,
            "config": {
                "d_model":     cfg.D_MODEL,
                "num_blocks":  cfg.NUM_BLOCKS,
                "num_classes": cfg.NUM_CLASSES,
            }
        }, ckpt_path)
        print(f"---> Đã lưu checkpoint: {ckpt_path}\n")


if __name__ == '__main__':
    train()