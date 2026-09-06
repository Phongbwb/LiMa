import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import v2

from models.classifier_trajectory import IntentPredictor


# ==========================================
# LABEL SMOOTHING LOSS — tránh model quá tự tin
# ==========================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, class_weights=None):
        super().__init__()
        self.smoothing     = smoothing
        self.num_classes   = num_classes
        self.class_weights = class_weights  # tensor (C,) hoặc None

    def forward(self, logits, targets):
        # Soft targets: (1-ε) cho class đúng, ε/(C-1) cho class sai
        with torch.no_grad():
            smooth_val  = self.smoothing / (self.num_classes - 1)
            soft_target = torch.full_like(logits, smooth_val)
            soft_target.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_prob = F.log_softmax(logits, dim=1)          # (B, C)

        if self.class_weights is not None:
            # Trọng số theo class của từng sample
            w = self.class_weights[targets]               # (B,)
            loss = -(soft_target * log_prob).sum(dim=1)   # (B,)
            return (loss * w).mean()
        else:
            return -(soft_target * log_prob).sum(dim=1).mean()


# ==========================================
# CONFIG
# ==========================================
class ModelConfig:
    def __init__(self):
        self.TASK        = 'intent'
        self.NUM_CLASSES = 3
        self.JSON_PATH   = './data/json/direction_trainset_train.json'
        self.D_MODEL     = 256
        self.NUM_BLOCKS  = 3


# ==========================================
# PAIRED TRANSFORM
# ==========================================
class PairedColorTransform:
    def __init__(self, size=224, is_minority=False):
        self.size = size

        if is_minority:
            self.spatial_crop = v2.Compose([
                v2.Resize((256, 256)),
                v2.RandomCrop(size),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=15),
                v2.RandomPerspective(distortion_scale=0.2, p=0.5),
            ])
            self.spatial_full = v2.Compose([
                v2.Resize((256, 256)),
                v2.RandomCrop(size),
                v2.RandomHorizontalFlip(p=0.5),
            ])
            self.color = v2.Compose([
                v2.RandomApply([v2.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15
                )], p=0.9),
                v2.RandomGrayscale(p=0.05),
            ])
        else:
            self.spatial_crop = v2.Compose([
                v2.Resize((size, size)),
                v2.RandomHorizontalFlip(p=0.5),
            ])
            self.spatial_full = v2.Compose([
                v2.Resize((size, size)),
                v2.RandomHorizontalFlip(p=0.5),
            ])
            self.color = v2.Compose([
                v2.RandomApply([v2.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                )], p=0.8),
            ])

        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
        ])

    def __call__(self, crop_img, full_img):
        crop = self.spatial_crop(crop_img)
        full = self.spatial_full(full_img)

        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed % (2**31))
        crop = self.color(crop)
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed % (2**31))
        full = self.color(full)

        return self.to_tensor(crop), self.to_tensor(full)


# ==========================================
# DATASET — undersample cls0 để cân bằng
# ==========================================
class IntentDataset(Dataset):
    def __init__(self, cfg, base_dir, undersample_ratio=5.0):
        """
        undersample_ratio: cls0 được giữ lại tối đa ratio × max(cls1, cls2).
        Ví dụ ratio=5, cls1=256, cls2=116 → cls0 giữ tối đa 5×256=1280 mẫu.
        Còn lại cls1, cls2 giữ nguyên 100%.
        """
        self.cfg          = cfg
        self.base_dir     = base_dir
        self.tfm_majority = PairedColorTransform(size=224, is_minority=False)
        self.tfm_minority = PairedColorTransform(size=224, is_minority=True)

        # --- Đọc toàn bộ JSON ---
        print(f"====> Đang đọc dữ liệu từ: {cfg.JSON_PATH}...")
        buckets = {0: [], 1: [], 2: []}   # gom theo class
        with open(cfg.JSON_PATH, 'r') as f:
            raw_data = json.load(f)

        for key, item in raw_data.items():
            label = item["id"]
            if label == 3 or label not in buckets:
                continue
            img_path = os.path.join(base_dir, item["frames"])
            if os.path.exists(img_path):
                buckets[label].append({
                    "full_path": img_path,
                    "boxes":     item["boxes"],
                    "label":     label
                })

        counts_raw = {c: len(v) for c, v in buckets.items()}
        print(f"      Phân bố gốc  : {counts_raw}")

        # --- Undersample cls0 ---
        minority_max  = max(len(buckets[1]), len(buckets[2]))
        cls0_cap      = int(minority_max * undersample_ratio)
        if len(buckets[0]) > cls0_cap:
            buckets[0] = random.sample(buckets[0], cls0_cap)

        counts_after = {c: len(v) for c, v in buckets.items()}
        print(f"      Sau undersample (ratio={undersample_ratio}×): {counts_after}")

        # --- Ghép lại và shuffle ---
        self.data = buckets[0] + buckets[1] + buckets[2]
        random.shuffle(self.data)
        self.labels_list = [d["label"] for d in self.data]

        print(f"      Tổng mẫu dùng để train: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def get_class_weights(self, device):
        """Sqrt-inverse weights — nhẹ nhàng, tránh quá cực đoan."""
        counts  = np.bincount(self.labels_list, minlength=self.cfg.NUM_CLASSES)
        weights = 1.0 / np.sqrt(counts + 1e-5)
        weights = weights / weights.sum() * self.cfg.NUM_CLASSES
        return torch.FloatTensor(weights).to(device)

    def __getitem__(self, idx):
        item  = self.data[idx]
        label = item["label"]

        full_image = Image.open(item["full_path"]).convert("RGB")
        x, y, w, h = item["boxes"]
        crop_image  = full_image.crop((x, y, x + w, y + h))

        tfm = self.tfm_minority if label in (1, 2) else self.tfm_majority
        crop_tensor, full_tensor = tfm(crop_image, full_image)

        return (crop_tensor, full_tensor), torch.tensor(label, dtype=torch.long)


# ==========================================
# TRAIN
# ==========================================
def train():
    BASE_DIR          = 'data/cityflownl/data'
    BATCH_SIZE        = 32
    EPOCHS            = 100
    LEARNING_RATE     = 1e-5
    UNDERSAMPLE_RATIO = 3.0   # cls0 ≤ 3 × max(cls1, cls2) = 3×256 = 768
    SMOOTHING         = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = ModelConfig()

    print(f"\n{'='*50}")
    print(f"  THIẾT BỊ         : {device}")
    print(f"  TASK             : {cfg.TASK.upper()}")
    print(f"  UNDERSAMPLE RATIO: {UNDERSAMPLE_RATIO}×")
    print(f"  LABEL SMOOTHING  : {SMOOTHING}")
    print(f"{'='*50}\n")

    dataset = IntentDataset(cfg, base_dir=BASE_DIR,
                            undersample_ratio=UNDERSAMPLE_RATIO)
    if len(dataset) == 0:
        print("❌ Không có dữ liệu. Dừng."); return

    # Shuffle thật sự mỗi epoch (không dùng Sampler)
    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True
    )

    model = IntentPredictor(d_model=cfg.D_MODEL, num_blocks=cfg.NUM_BLOCKS).to(device)

    class_weights = dataset.get_class_weights(device)
    print(f"  Loss class weights: {class_weights.cpu().numpy().round(3)}")
    criterion = LabelSmoothingLoss(
        num_classes   = cfg.NUM_CLASSES,
        smoothing     = SMOOTHING,
        class_weights = class_weights
    )

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    os.makedirs('./checkpoints/intent/', exist_ok=True)
    best_bal_acc = 0.0

    for epoch in range(EPOCHS):
        # Reshuffle cls0 mỗi epoch — mỗi epoch thấy tập con cls0 khác nhau
        dataset.data = (
            random.sample([d for d in dataset.data if d["label"] == 0],
                          min(int(max(sum(1 for d in dataset.data if d["label"] == 1),
                                      sum(1 for d in dataset.data if d["label"] == 2))
                                  * UNDERSAMPLE_RATIO),
                              sum(1 for d in dataset.data if d["label"] == 0)))
            + [d for d in dataset.data if d["label"] in (1, 2)]
        )
        random.shuffle(dataset.data)
        dataset.labels_list = [d["label"] for d in dataset.data]

        model.train()
        running_loss      = 0.0
        correct_per_class = np.zeros(cfg.NUM_CLASSES)
        total_per_class   = np.zeros(cfg.NUM_CLASSES)

        for i, (inputs, labels) in enumerate(dataloader):
            crop_inputs, full_inputs = inputs
            crop_inputs = crop_inputs.to(device)
            full_inputs = full_inputs.to(device)
            labels      = labels.to(device)

            optimizer.zero_grad()
            out    = model(crop_inputs, full_inputs)
            logits = out["logits"]
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(logits, 1)

            for cls in range(cfg.NUM_CLASSES):
                mask = labels == cls
                correct_per_class[cls] += (preds[mask] == labels[mask]).sum().item()
                total_per_class[cls]   += mask.sum().item()

            if (i + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS}] "
                      f"Batch [{i+1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f}")

        scheduler.step()

        epoch_loss   = running_loss / len(dataloader)
        per_cls_acc  = [
            correct_per_class[c] / max(total_per_class[c], 1) * 100
            for c in range(cfg.NUM_CLASSES)
        ]
        balanced_acc = float(np.mean(per_cls_acc))
        overall_acc  = correct_per_class.sum() / total_per_class.sum() * 100

        per_cls_str = " | ".join(
            f"cls{c}: {per_cls_acc[c]:.1f}% ({int(total_per_class[c])})"
            for c in range(cfg.NUM_CLASSES)
        )

        print(f"\n---> EPOCH {epoch+1:02d} | "
              f"LOSS: {epoch_loss:.4f} | "
              f"ACC: {overall_acc:.2f}% | "
              f"BAL_ACC: {balanced_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"     Per-class → {per_cls_str}")

        ckpt_path = f'./checkpoints/intent/intent_ep{epoch+1}.pth'
        torch.save(model.state_dict(), ckpt_path)

        if balanced_acc > best_bal_acc:
            best_bal_acc = balanced_acc
            torch.save(model.state_dict(), './checkpoints/intent/intent_best.pth')
            print(f"     ✅ Best model (BAL_ACC={best_bal_acc:.2f}%)")

        print()

    print(f"🏁 Xong! Best Balanced ACC: {best_bal_acc:.2f}%")


if __name__ == '__main__':
    train()