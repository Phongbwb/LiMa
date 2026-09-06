import os
import json
import torch
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import Counter, defaultdict
from torch.cuda.amp import autocast

# Model mới chỉ nhận crop_img (bỏ nhánh context/full_img)
from models.classifier_type import IntentPredictor


def get_sort_key(f_path):
    nums = re.findall(r'\d+', str(f_path))
    return (int(nums[-1]), f_path) if nums else (0, f_path)


# ==========================================
# 1. DATASET: MỖI SAMPLE = 1 ẢNH ĐƠN LẺ
#    Bỏ full_tensor — model mới không dùng
# ==========================================
class VehicleTypeInferDataset(Dataset):
    def __init__(self, json_path, base_dir, transform=None):
        self.transform = transform
        self.base_dir  = base_dir
        self.data      = []   # mỗi phần tử: {"uuid", "frame_path", "box"}

        print(f"====> Đang đọc dữ liệu từ: {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        for uuid_key, item in raw_data.items():
            raw_frames = item["frames"]
            raw_boxes  = item["boxes"]
            assert len(raw_frames) == len(raw_boxes), \
                f"UUID {uuid_key}: số frames ({len(raw_frames)}) ≠ boxes ({len(raw_boxes)})"

            # Sắp xếp theo thứ tự thời gian
            combined = sorted(
                zip(raw_frames, raw_boxes),
                key=lambda x: get_sort_key(x[0])
            )
            for frame_path, box in combined:
                self.data.append({
                    "uuid":       uuid_key,
                    "frame_path": frame_path,
                    "box":        box          # [x, y, w, h]
                })

        print(f"====> Hoàn tất! Tổng số ảnh: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item      = self.data[idx]
        full_path = os.path.join(self.base_dir, item["frame_path"])
        uuid_key  = item["uuid"]

        try:
            full_img     = Image.open(full_path).convert("RGB")
            img_w, img_h = full_img.size

            x, y, w, h = item["box"]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, int(x + w))
            y2 = min(img_h, int(y + h))

            # Bảo vệ trường hợp box rỗng/không hợp lệ
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Box không hợp lệ: [{x},{y},{w},{h}]")

            crop_img = full_img.crop((x1, y1, x2, y2))

        except Exception as e:
            print(f"⚠️ Lỗi đọc ảnh {full_path}: {e}")
            dummy = torch.zeros(3, 224, 224)
            return dummy, uuid_key

        if self.transform:
            crop_tensor = self.transform(crop_img)
        else:
            crop_tensor = transforms.ToTensor()(crop_img)

        return crop_tensor, uuid_key


# ==========================================
# 2. HÀM INFERENCE
# ==========================================
def infer_type():
    BASE_DIR         = 'data/cityflownl/data'
    TEST_JSON_PATH   = './data/json/test-tracks.json'
    WEIGHTS_PATH     = './checkpoints/type/type_efficientnet_mamba_ep06.pth'
    OUTPUT_JSON_PATH = './data/json/test_tracks_type.json'

    D_MODEL     = 256
    NUM_BLOCKS  = 3
    NUM_CLASSES = 7
    BATCH_SIZE  = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*40}")
    print(f"  TASK    : TYPE ({NUM_CLASSES} classes)")
    print(f"  DEVICE  : {device}")
    print(f"  WEIGHTS : {WEIGHTS_PATH}")
    print(f"{'='*40}\n")

    # ------------------------------------------
    # TRANSFORM — giống train nhưng không augment
    # Dùng CenterCrop thay Resize thẳng để giữ tỉ lệ
    # ------------------------------------------
    IMG_SIZE = 224
    test_transforms = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.12)),   # +12% margin, giống train
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])

    # ------------------------------------------
    # DATASET & DATALOADER
    # ------------------------------------------
    dataset = VehicleTypeInferDataset(
        TEST_JSON_PATH, base_dir=BASE_DIR, transform=test_transforms
    )
    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True
    )

    # ------------------------------------------
    # MODEL — load từ checkpoint
    # ------------------------------------------
    model = IntentPredictor(
        d_model     = D_MODEL,
        num_blocks  = NUM_BLOCKS,
        num_classes = NUM_CLASSES,
    ).to(device)

    print(f"====> Tải weights từ: {WEIGHTS_PATH}")
    ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)

    # Hỗ trợ cả 2 định dạng checkpoint:
    #   - dict đầy đủ {"model_state": ..., "epoch": ..., ...}  (checkpoint mới)
    #   - state_dict thuần                                      (checkpoint cũ)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        print(f"   └─ Epoch: {ckpt.get('epoch', '?')} | "
              f"Acc: {ckpt.get('acc', 0):.2f}%")
    else:
        model.load_state_dict(ckpt)

    model.eval()

    # ------------------------------------------
    # DỰ ĐOÁN TỪNG ẢNH, GOM THEO UUID
    # ------------------------------------------
    uuid_predictions = defaultdict(list)   # uuid -> [pred, pred, ...]

    print("\n====> Đang chạy dự đoán...")
    with torch.no_grad():
        for crop_tensors, uuid_keys in tqdm(dataloader, desc="Inference"):
            crop_tensors = crop_tensors.to(device, non_blocking=True)

            with autocast():
                # Model mới: forward(crop_img) → dict{"logits", "features", "raw_features"}
                outputs = model(crop_tensors)
                logits  = outputs["logits"]       # (B, NUM_CLASSES)

            preds = logits.argmax(dim=1).cpu().tolist()  # (B,)

            for uuid, pred in zip(uuid_keys, preds):
                uuid_predictions[uuid].append(pred)

    # ------------------------------------------
    # MAJORITY VOTE THEO UUID
    # ------------------------------------------
    print("\n====> Tiến hành Majority Vote...")
    final_results = {}
    vote_summary  = Counter()   # thống kê phân phối kết quả

    for uuid, preds_list in uuid_predictions.items():
        vote_counts = Counter(preds_list)
        best_id     = vote_counts.most_common(1)[0][0]
        confidence  = vote_counts[best_id] / len(preds_list)  # tỉ lệ đồng thuận

        final_results[uuid] = {
            "id":         best_id,
            "confidence": round(confidence, 4),
            "votes":      dict(vote_counts),
            "n_frames":   len(preds_list),
        }
        vote_summary[best_id] += 1

    # In phân phối kết quả
    print("\n  Phân phối kết quả dự đoán:")
    total_uuids = len(final_results)
    for cls_id in sorted(vote_summary):
        count = vote_summary[cls_id]
        bar   = '█' * int(count / total_uuids * 30)
        print(f"    Class {cls_id}: {count:5d} UUID  {bar}")

    # ------------------------------------------
    # LƯU JSON
    # ------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ HOÀN TẤT! Lưu tại: {OUTPUT_JSON_PATH}")
    print(f"   Tổng UUID: {total_uuids}")


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    infer_type()