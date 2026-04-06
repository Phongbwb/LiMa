import os
import json
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class CityFlowNLDataset(Dataset):
    def __init__(self, data_root, json_path, num_frames=8, img_size=112, is_train=True):
        """
        Dataset cho bài toán Joint Detection & Tracking với Text Query (CityFlow-NL).
        
        Args:
            data_root (str): Thư mục gốc chứa video/frames toàn cảnh (VD: ./data/cityflow-nl/train)
            json_path (str): Đường dẫn tới file train-tracks.json
            num_frames (int): Số lượng frame trích xuất cho mỗi track
            img_size (int): Kích thước ảnh đầu vào của mạng Li-Ma
            is_train (bool): Bật chế độ Data Augmentation & Negative Sampling nếu đang train
        """
        super().__init__()
        self.data_root = data_root
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_train = is_train
        
        # 1. Đọc dữ liệu từ file JSON
        print(f"Đang tải dữ liệu từ {json_path}...")
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
        # 2. Làm phẳng (Flatten) dữ liệu
        # Mỗi chiếc xe có 3 câu miêu tả. Ta tách chúng ra thành các mẫu độc lập.
        self.samples = []
        for track_id, track_info in raw_data.items():
            for nl_query in track_info['nl']:
                self.samples.append({
                    'track_id': track_id,
                    'frames': track_info['frames'], # Danh sách đường dẫn frame tương đối
                    'boxes': track_info['boxes'],   # Danh sách box tương ứng từng frame
                    'text': nl_query
                })
        print(f"Đã tải xong {len(self.samples)} mẫu dữ liệu video-text.")

        # 3. Khởi tạo phép biến đổi ảnh (Transforms)
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # THÊM DÒNG NÀY VÀO CUỐI HÀM __init__:
        # Tải bộ từ điển CLIP Embeddings đã trích xuất
        emb_path = os.path.join(data_root, 'clip_text_embeddings.pt')
        print("Đang tải CLIP Text Embeddings...")
        self.text_embs_dict = torch.load(emb_path)

    def __len__(self):
        return len(self.samples)

    def _get_text_embedding(self, text):
        
        return self.text_embs_dict[text].clone()

    def _normalize_bbox(self, box, orig_w, orig_h):
        """
        Chuyển box từ hệ [top_left_x, top_left_y, w, h] (pixel)
        Sang hệ chuẩn hóa YOLO [center_x, center_y, norm_w, norm_h] (0 -> 1)
        """
        tl_x, tl_y, w, h = box
        
        # Tính tâm box
        c_x = tl_x + (w / 2.0)
        c_y = tl_y + (h / 2.0)
        
        # Chuẩn hóa về khoảng [0, 1]
        norm_c_x = c_x / orig_w
        norm_c_y = c_y / orig_h
        norm_w = w / orig_w
        norm_h = h / orig_h
        
        # Tránh các giá trị văng ra khỏi khung hình
        norm_c_x = np.clip(norm_c_x, 0, 1)
        norm_c_y = np.clip(norm_c_y, 0, 1)
        norm_w = np.clip(norm_w, 0, 1)
        norm_h = np.clip(norm_h, 0, 1)
        
        return [norm_c_x, norm_c_y, norm_w, norm_h]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # --- NEGATIVE SAMPLING (Tạo nhãn is_match) ---
        is_match = 1.0
        target_sample = sample
        
        # Tỉ lệ 50% lấy mẫu sai (Negative) để dạy mạng biết cách "Mù" (loss_gate)
        if self.is_train and random.random() < 0.5:
            is_match = 0.0
            # Lấy ngẫu nhiên một video track khác
            neg_idx = random.randint(0, len(self.samples) - 1)
            while self.samples[neg_idx]['track_id'] == sample['track_id']:
                neg_idx = random.randint(0, len(self.samples) - 1)
            
            # Thay đổi video đầu vào thành video sai, nhưng GIỮ NGUYÊN text truy vấn
            target_sample = self.samples[neg_idx]

        # --- XỬ LÝ VIDEO & BBOX ---
        frames_paths = target_sample['frames']
        boxes = target_sample['boxes']
        total_frames = len(frames_paths)

        # Trích xuất đều `num_frames` từ toàn bộ track
        indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        
        video_tensor = []
        target_box = [0.0, 0.0, 0.0, 0.0]

        for i, frame_idx in enumerate(indices):
            # Đường dẫn vật lý tới ảnh Full Frame
            img_path = os.path.join(self.data_root, frames_paths[frame_idx])
            
            # Đọc ảnh
            img = cv2.imread(img_path)
            if img is None:
                # Fallback nếu đường dẫn lỗi (Tạo ảnh đen)
                img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            orig_h, orig_w, _ = img.shape
            
            # Chuyển qua PIL để dùng torchvision transforms
            img_pil = Image.fromarray(img)
            img_tensor = self.transform(img_pil) # [C, H, W]
            video_tensor.append(img_tensor)
            
            # Lấy Bounding Box của frame ở TRUNG TÂM đoạn video làm nhãn đại diện
            # (Hoặc bạn có thể trả về mảng T boxes nếu mạng Li-Ma xuất ra box theo từng frame)
            if i == self.num_frames // 2:
                if is_match == 1.0:
                    raw_box = boxes[frame_idx]
                    target_box = self._normalize_bbox(raw_box, orig_w, orig_h)

        # Chồng các frame lại thành shape [C, T, H, W]
        video_tensor = torch.stack(video_tensor, dim=1)
        
        # --- XỬ LÝ TEXT ---
        # Text gốc của sample, KHÔNG PHẢI của target_sample
        text_emb = self._get_text_embedding(sample['text'])

        # --- PSEUDO LABELS ---
        # Bộ dữ liệu gốc không có nhãn phân cấp, gán mặc định để chống lỗi hàm loss cũ
        coarse_label = 0 
        fine_label = 0

        return {
            'video': video_tensor,                  # [3, T, 112, 112]
            'text_emb': text_emb,                   # [d_text]
            'is_match': torch.tensor(is_match, dtype=torch.float32), # [1]
            'bbox': torch.tensor(target_box, dtype=torch.float32),   # [4]
            'coarse_label': torch.tensor(coarse_label, dtype=torch.long),
            'fine_label': torch.tensor(fine_label, dtype=torch.long)
        }