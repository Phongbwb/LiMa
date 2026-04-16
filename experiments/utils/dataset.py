import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CityFlowNLDataset(Dataset):
    def __init__(self, json_path, data_root, text_emb_path, transform=None, 
                 max_frames=8, img_size=384, feat_stride=16):
        """
        Args:
            json_path: Đường dẫn file train-track.json hoặc test-track.json
            data_root: Thư mục chứa các folder S01, S02...
            max_frames: Số lượng frame tối đa load trong 1 batch (T) để tránh OOM
            feat_stride: Tỉ lệ thu nhỏ của backbone (xuống heatmap)
        """
        self.data_root = data_root
        self.json_path = json_path # <--- Sửa lỗi AttributeError
        
        with open(json_path, 'r') as f:
            full_data = json.load(f)
        
        # Chỉ nạp những folder bạn đang có (S01, S02, S05)
        available_folders = ["S01", "S02", "S05"]
        self.data = {}
        for tid, info in full_data.items():
            f_path = info['frames'][0]
            if any(folder in f_path for folder in available_folders):
                self.data[tid] = info
        
        # Danh sách ID này bây giờ CHỈ chứa S01 (đối với tập test)
        self.track_ids = list(self.data.keys())
        print(f"📦 Dataset: Đã nạp {len(self.track_ids)} tracks.")
        
        self.max_frames = max_frames
        self.img_size = img_size
        self.feat_stride = feat_stride
        self.output_size = img_size // feat_stride
        
        # Transform cơ bản: Resize -> Tensor -> Normalize (theo chuẩn ImageNet)
        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 1. TẢI TỪ ĐIỂN TEXT EMBEDDING VÀO RAM
        print(f"Đang tải Text Embeddings từ {text_emb_path}...")
        self.text_embs = torch.load(text_emb_path)

    def __len__(self):
        return len(self.track_ids)

    def _get_gaussian_radius(self, box_w, box_h, min_overlap=0.7):
        """Tính bán kính Gaussian cho Heatmap dựa trên kích thước box"""
        a1  = 1
        b1  = (box_w + box_h)
        c1  = box_w * box_h * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (box_w + box_h)
        c2  = (1 - min_overlap) * box_w * box_h
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (box_w + box_h)
        c3  = (min_overlap - 1) * box_w * box_h
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def _draw_gaussian(self, heatmap, center, radius, k=1):
        """Vẽ điểm cực đại Gaussian lên heatmap"""
        diameter = 2 * radius + 1
        gaussian = self._gaussian_label(radius, sigma=diameter/6)
        
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]
        
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def _gaussian_label(self, radius, sigma=1):
        x, y = np.ogrid[-radius:radius+1, -radius:radius+1]
        h = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        item = self.data[track_id]
        
        frames_path = item['frames']
        boxes = np.array(item['boxes']) # [x, y, w, h]
        n_frames = len(frames_path)

        # --- ĐOẠN CODE MỚI: XỬ LÝ SỐ LƯỢNG FRAME ---
        indices = list(range(n_frames))
        
        if n_frames > self.max_frames:
            # Video dài: Cắt một cửa sổ ngẫu nhiên
            start_f = np.random.randint(0, n_frames - self.max_frames)
            indices = indices[start_f : start_f + self.max_frames]
        else:
            # Video ngắn: Lặp lại frame cuối cùng cho đến khi đủ max_frames
            while len(indices) < self.max_frames:
                indices.append(indices[-1])
        
        # Áp dụng indices đã xử lý để lấy frame và box
        selected_frames = [frames_path[i] for i in indices]
        selected_boxes = [boxes[i] for i in indices]

        video_tensor = []
        hms, szs, offs = [], [], []

        # FIX: Xử lý đường dẫn cho frame đầu tiên để lấy kích thước
        first_frame_clean = selected_frames[0].lstrip('./') 
        first_frame_full = os.path.normpath(os.path.join(self.data_root, first_frame_clean))
        
        with Image.open(first_frame_full) as img:
            orig_w, orig_h = img.size

        for i, f_path in enumerate(selected_frames):
            full_path = os.path.join(self.data_root, f_path)
            img = Image.open(full_path).convert('RGB')
            video_tensor.append(self.transform(img))

            # Ground Truth cho CenterNet
            hm = np.zeros((self.output_size, self.output_size), dtype=np.float32)
            sz = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
            off = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)

            box = selected_boxes[i] # [x, y, w, h]
            # Chuyển sang tọa độ Feature Map
            f_box_w = (box[2] / orig_w) * self.output_size
            f_box_h = (box[3] / orig_h) * self.output_size
            f_cx = ((box[0] + box[2]/2) / orig_w) * self.output_size
            f_cy = ((box[1] + box[3]/2) / orig_h) * self.output_size

            radius = max(0, int(self._get_gaussian_radius(f_box_w, f_box_h)))
            ct = np.array([f_cx, f_cy], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            
            # Vẽ Heatmap
            self._draw_gaussian(hm, ct_int, radius)
            # Ghi nhận Size (w, h) và Offset tại vị trí tâm
            if ct_int[0] < self.output_size and ct_int[1] < self.output_size:
                sz[:, ct_int[1], ct_int[0]] = [f_box_w, f_box_h]
                off[:, ct_int[1], ct_int[0]] = ct - ct_int

            hms.append(hm); szs.append(sz); offs.append(off)

        # 4. Xử lý Text (NL)
        # Chọn ngẫu nhiên 1 trong các câu mô tả để tăng tính đa dạng (Augmentation)
        # --- BẢN VÁ TRIỆT ĐỂ: XỬ LÝ TEXT ---
        all_nl = item.get('nl', []) + item.get('nl_other_views', [])
        text_tensor = torch.zeros(32, 512, dtype=torch.float32)

        # 1. Lọc bỏ ngay lập tức những chuỗi rỗng hoặc chỉ có dấu cách
        valid_nls = [text for text in all_nl if text.strip() != ""]

        # 2. Nếu sau khi lọc mà vẫn còn text hợp lệ thì mới xử lý
        if len(valid_nls) > 0:
            selected_nl = np.random.choice(valid_nls).strip().lower()
            
            if selected_nl in self.text_embs:
                text_tensor = self.text_embs[selected_nl]
            else:
                # Lúc này chắc chắn selected_nl là một câu có chữ đàng hoàng
                print(f"⚠️ Cảnh báo: Không tìm thấy embedding cho '{selected_nl}'")

        return {
            "video": torch.stack(video_tensor), # [T, 3, 384, 384]
            "hm": torch.from_numpy(np.stack(hms)),   # [T, 1, H', W']
            "sz": torch.from_numpy(np.stack(szs)),   # [T, 2, H', W']
            "off": torch.from_numpy(np.stack(offs)), # [T, 2, H', W']
            "text_tokens": text_tensor,
            "track_id": track_id
        }