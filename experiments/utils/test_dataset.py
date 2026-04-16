import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageFilter

class CityFlowNLDataset(Dataset):
    def __init__(self, data_root, json_path, num_frames=8, img_size=384, down_ratio=4, is_train=True, use_motion=True):
        super().__init__()
        self.data_root = data_root
        self.num_frames = num_frames
        self.img_size = img_size
        self.down_ratio = down_ratio # Heatmap sẽ nhỏ hơn ảnh gốc (vd: 256/4 = 64)
        self.is_train = is_train
        self.use_motion = use_motion

        with open(json_path, 'r') as f:
            raw_data = json.load(f)

        self.samples = []
        self.camera_index = {}
        for track_id, track_info in raw_data.items():
            frames = track_info["frames"]
            cam_id = frames[0].split('/')[3]
            for text in track_info["nl"] + track_info.get("nl_other_views", []):
                self.samples.append({
                    "track_id": track_id, "frames": frames, 
                    "boxes": track_info["boxes"], "text": text, "cam_id": cam_id
                })
                if cam_id not in self.camera_index:
                    self.camera_index[cam_id] = []
                self.camera_index[cam_id].append(len(self.samples) - 1)

        self.text_embs = torch.load(os.path.join(data_root, "clip_text_embeddings.pt"))
        
        self.base_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)
    
    # ==========================================
    # CÁC HÀM TIỆN ÍCH HỖ TRỢ
    # ==========================================
    def _normalize_bbox(self, box, w, h):
        x, y, bw, bh = box
        return [
            np.clip((x + bw / 2) / w, 0, 1), np.clip((y + bh / 2) / h, 0, 1),
            np.clip(bw / w, 0, 1), np.clip(bh / h, 0, 1)
        ]

    def _get_text_emb(self, text):
        clean_text = text.strip().lower()
        if clean_text in self.text_embs: 
            return self.text_embs[clean_text].clone()
        # [FIX]: Ngăn chặn NaN cho Cosine Gate bằng nhiễu siêu nhỏ
        dim = next(iter(self.text_embs.values())).shape[0]
        return torch.randn(dim) * 1e-5

    def _sample_indices(self, total_frames):
        T_frames = self.num_frames
        if total_frames >= T_frames:
            start = random.randint(0, total_frames - T_frames)
            return list(range(start, start + T_frames)), [1] * T_frames
        else:
            indices, mask = list(range(total_frames)), [1] * total_frames
            while len(indices) < T_frames:
                indices.append(indices[-1])
                mask.append(0)
            return indices, mask

    def _get_hard_negative(self, sample):
        # [FIX]: Lấy mẫu Negative tất định, chống kẹt lặp vô hạn
        candidates = self.camera_index[sample["cam_id"]]
        valid_candidates = [idx for idx in candidates if self.samples[idx]["track_id"] != sample["track_id"]]
        
        if len(valid_candidates) > 0:
            return self.samples[random.choice(valid_candidates)]
        
        while True:
            neg_idx = random.randint(0, len(self.samples) - 1)
            if self.samples[neg_idx]["track_id"] != sample["track_id"]:
                return self.samples[neg_idx]

    # ==========================================
    # CÁC HÀM VẼ GAUSSIAN HEATMAP CHO CENTER-POINT
    # ==========================================
    def _gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def _draw_umich_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self._gaussian2D((diameter, diameter), sigma=diameter / 6)
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    # ==========================================
    # HÀM CHÍNH: LẤY DỮ LIỆU ĐỂ HUẤN LUYỆN
    # ==========================================
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text_emb = self._get_text_emb(sample["text"])

        is_match = 1.0
        video_sample = sample

        # Sinh 50% Negative Sample để dạy Cổng (Gate) cách Đóng
        if self.is_train and random.random() < 0.5:
            is_match = 0.0
            video_sample = self._get_hard_negative(sample)

        frames = video_sample["frames"]
        boxes = video_sample["boxes"]
        indices, mask = self._sample_indices(len(frames))

        video = []
        bbox_seq = []
        
        # Khởi tạo ma trận Heatmap, Size và Offset
        hm_size = self.img_size // self.down_ratio
        heatmap_seq = np.zeros((self.num_frames, 1, hm_size, hm_size), dtype=np.float32)
        size_seq = np.zeros((self.num_frames, 2, hm_size, hm_size), dtype=np.float32)
        offset_seq = np.zeros((self.num_frames, 2, hm_size, hm_size), dtype=np.float32)

        # [FIX]: Sinh tham số Augmentation ĐỒNG BỘ 1 lần cho cả chuỗi Frame
        do_flip = self.is_train and (random.random() < 0.5)
        do_blur = self.is_train and (random.random() < 0.2)
        do_color = self.is_train and (random.random() < 0.8)

        color_ops = []
        if do_color:
            b_factor = random.uniform(0.7, 1.3)
            c_factor = random.uniform(0.7, 1.3)
            s_factor = random.uniform(0.7, 1.3)
            h_factor = random.uniform(-0.1, 0.1)
            color_ops = [
                lambda img: T.functional.adjust_brightness(img, b_factor),
                lambda img: T.functional.adjust_contrast(img, c_factor),
                lambda img: T.functional.adjust_saturation(img, s_factor),
                lambda img: T.functional.adjust_hue(img, h_factor)
            ]
            random.shuffle(color_ops)

        # Duyệt qua từng khung hình
        for i_idx, i in enumerate(indices):
            img_path = os.path.join(self.data_root, frames[i])
            img = Image.open(img_path).convert('RGB')
            w_orig, h_orig = img.size

            # Áp dụng Augmentation Đồng bộ
            for op in color_ops: img = op(img)
            if do_blur: img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
            if do_flip: img = T.functional.hflip(img)

            img_tensor = self.base_transform(img)
            video.append(img_tensor)

            if is_match:
                cx, cy, bw, bh = self._normalize_bbox(boxes[i], w_orig, h_orig)
                if do_flip: cx = 1.0 - cx # Lật tọa độ X
                
                bbox_seq.append([cx, cy, bw, bh])
                
                # --- VẼ CENTER-POINT TARGETS ---
                # Chuyển đổi tọa độ [0, 1] sang tọa độ lưới Heatmap
                center_x = cx * hm_size
                center_y = cy * hm_size
                
                # Tính bán kính Gaussian (Dựa trên diện tích box trên lưới)
                box_w_grid = bw * hm_size
                box_h_grid = bh * hm_size
                radius = max(1, int(np.sqrt(box_w_grid * box_h_grid) * 0.15)) # Hệ số 0.15 tùy chỉnh
                
                heatmap_seq[i_idx, 0] = self._draw_umich_gaussian(
                    heatmap_seq[i_idx, 0], (center_x, center_y), radius
                )
                
                # Lưu kích thước (Dùng Logarit để chống trôi Gradient)
                x_int, y_int = int(center_x), int(center_y)
                if 0 <= x_int < hm_size and 0 <= y_int < hm_size:
                    size_seq[i_idx, 0, y_int, x_int] = np.log(bw * hm_size + 1e-6)
                    size_seq[i_idx, 1, y_int, x_int] = np.log(bh * hm_size + 1e-6)
                    
                    # Lưu độ lệch tâm (Offset)
                    offset_seq[i_idx, 0, y_int, x_int] = center_x - x_int
                    offset_seq[i_idx, 1, y_int, x_int] = center_y - y_int
            else:
                bbox_seq.append([0.0, 0.0, 0.0, 0.0])

        video = torch.stack(video, dim=1)           # Shape: [C, T, H, W]
        bbox_seq = torch.tensor(bbox_seq).float()   # Shape: [T, 4]
        mask = torch.tensor(mask).float()           # Shape: [T]

        # Convert numpy arrays to tensors
        heatmap_seq = torch.from_numpy(heatmap_seq) # [T, 1, H, W]
        size_seq = torch.from_numpy(size_seq)       # [T, 2, H, W]
        offset_seq = torch.from_numpy(offset_seq)   # [T, 2, H, W]

        if self.use_motion:
            motion = bbox_seq[1:] - bbox_seq[:-1]   # Shape: [T-1, 4]
        else:
            motion = torch.zeros_like(bbox_seq[:-1])

        return {
            "video": video,
            "text_emb": text_emb,
            "heatmap": heatmap_seq, # TARGET QUAN TRỌNG NHẤT
            "size_map": size_seq,   # TARGET CHO REGRESSION SIZE
            "offset_map": offset_seq, # TARGET CHO REGRESSION OFFSET
            "bbox_seq": bbox_seq,
            "motion": motion,
            "mask": mask,
            "is_match": torch.tensor(is_match).float(),
            "track_id": sample["track_id"]
        }