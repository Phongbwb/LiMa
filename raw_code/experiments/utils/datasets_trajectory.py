import os
import json
import cv2
import random
import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset


# ==========================================
# 1. CÁC HÀM TIỆN ÍCH & AUGMENTATION
# ==========================================

def normalize_path(path_str):
    """Chuẩn hóa đường dẫn để tránh lỗi do hệ điều hành (Windows/Linux)"""
    return os.path.normpath(path_str).replace('\\', '/').lstrip('./')

def apply_bbox_jitter(boxes, img_w, img_h, jitter_ratio=0.05):
    """Thêm nhiễu ngẫu nhiên vào Bounding Box (BBox Jittering)"""
    if random.random() > 0.5:
        return boxes

    T = boxes.shape[0]
    noisy_boxes = boxes.clone()
    
    # Sinh nhiễu (noise)
    noise_x = (torch.rand(T) * 2 - 1) * jitter_ratio * boxes[:, 2]
    noise_y = (torch.rand(T) * 2 - 1) * jitter_ratio * boxes[:, 3]
    noise_w = (torch.rand(T) * 2 - 1) * jitter_ratio * boxes[:, 2]
    noise_h = (torch.rand(T) * 2 - 1) * jitter_ratio * boxes[:, 3]
    
    noisy_boxes[:, 0] += noise_x
    noisy_boxes[:, 1] += noise_y
    noisy_boxes[:, 2] += noise_w
    noisy_boxes[:, 3] += noise_h
    
    # GIẢI QUYẾT LỖI TYPE ERROR CỦA TORCH.CLAMP
    # Sử dụng torch.minimum và torch.maximum để so sánh Tensor an toàn
    zero_tensor = torch.tensor(0.0, device=noisy_boxes.device)
    
    noisy_boxes[:, 0] = torch.maximum(zero_tensor, torch.minimum(noisy_boxes[:, 0], img_w - noisy_boxes[:, 2]))
    noisy_boxes[:, 1] = torch.maximum(zero_tensor, torch.minimum(noisy_boxes[:, 1], img_h - noisy_boxes[:, 3]))
    
    # Đối với w và h, do min/max đều là con số (int/float) nên dùng clamp bình thường
    noisy_boxes[:, 2] = torch.clamp(noisy_boxes[:, 2], 1, img_w)
    noisy_boxes[:, 3] = torch.clamp(noisy_boxes[:, 3], 1, img_h)
    
    return noisy_boxes

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (DATASET)
# ==========================================

class VehicleTrackDataset(Dataset):
    def __init__(self, tracks_dict, action_json_path, seq_len=6, is_train=True, img_w=1920, img_h=1080):
        self.seq_len = seq_len
        self.is_train = is_train
        self.img_w = img_w
        self.img_h = img_h
        
        # 1. Nạp dữ liệu từ Dictionary được truyền vào (Tránh Leakage)
        self.tracks_data = tracks_dict 
        
        with open(action_json_path, 'r') as f:
            self.actions_data = json.load(f)
            
        # 2. Map nhãn bằng đường dẫn chuẩn hóa
        self.frame_to_label = {}
        for key, val in self.actions_data.items():
            clean_p = normalize_path(val["frames"])
            self.frame_to_label[clean_p] = val["id"]
            
        # 3. Trượt cửa sổ (Sliding Window)
        self.samples = []
        for track_id, track_info in self.tracks_data.items():
            frames = track_info["frames"]
            boxes = track_info["boxes"]
            
            num_frames = len(frames)
            if num_frames < self.seq_len:
                continue
            stride = 3 if self.is_train else 1   
            for i in range(0, num_frames - self.seq_len + 1, stride):
                seq_frames = frames[i : i + self.seq_len]
                seq_boxes = boxes[i : i + self.seq_len]
                
                clean_last_p = normalize_path(seq_frames[-1])
                label = self.frame_to_label.get(clean_last_p, 0)
                
                self.samples.append({
                    "frames": seq_frames,
                    "boxes": seq_boxes,
                    "label": label
                })

        # 4. Pipeline Transform (Đã fix lỗi tên class và lỗi antialias)
        if self.is_train:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((224, 224), antialias=True), # Fix cảnh báo Resize
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # Fix tên class
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # Fix tên class
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
            ])
        else:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((224, 224), antialias=True), # Fix cảnh báo Resize
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _compute_9d_kinematics(self, bboxes):
        T = bboxes.shape[0]
        features = torch.zeros((T, 9), dtype=torch.float32)
        eps = 1e-6 
        
        x_min, y_min, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        x_c, y_c = x_min + w / 2, y_min + h / 2
        
        features[:, 0], features[:, 1] = x_c / self.img_w, y_c / self.img_h
        features[:, 2], features[:, 3] = w / self.img_w, h / self.img_h
        features[:, 4] = w / (h + eps)
        
        v_x, v_y = torch.zeros(T), torch.zeros(T)
        v_x[1:] = (x_c[1:] - x_c[:-1]) / (w[1:] + eps)
        v_y[1:] = (y_c[1:] - y_c[:-1]) / (h[1:] + eps)
        if T > 1: v_x[0], v_y[0] = v_x[1], v_y[1]
            
        features[:, 5], features[:, 6] = v_x, v_y
        
        a_x, a_y = torch.zeros(T), torch.zeros(T)
        a_x[1:] = v_x[1:] - v_x[:-1]
        a_y[1:] = v_y[1:] - v_y[:-1]
        if T > 1: a_x[0], a_y[0] = a_x[1], a_y[1]
            
        features[:, 7], features[:, 8] = a_x, a_y
        return features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        boxes_tensor = torch.tensor(sample["boxes"], dtype=torch.float32)
        
        if self.is_train:
            boxes_tensor = apply_bbox_jitter(boxes_tensor, self.img_w, self.img_h, jitter_ratio=0.05)
            
        kinematics_tensor = self._compute_9d_kinematics(boxes_tensor)
        
        images = []
        # Khai báo thư mục gốc chứa dữ liệu của bạn
        base_dir = "./data/cityflownl/data" 
        
        for i, frame_path in enumerate(sample["frames"]):
            # Xóa các ký tự './' hoặc '/' ở đầu chuỗi JSON để nối path an toàn
            clean_frame_path = frame_path.lstrip('./').lstrip('/')
            
            # Gộp thư mục gốc với đường dẫn ảnh
            real_path = os.path.join(base_dir, clean_frame_path)
            
            # Đọc ảnh từ đường dẫn thực tế
            img = cv2.imread(real_path)
            
            if img is None:
                print(f"\n[CẢNH BÁO] Không đọc được ảnh: {real_path}")
                crop_img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x_min, y_min, w, h = [int(v) for v in sample["boxes"][i]]
                crop_img = img[max(0, y_min):y_min+h, max(0, x_min):x_min+w]
                if crop_img.size == 0:
                    crop_img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            crop_tensor = self.transform(crop_img) 
            images.append(crop_tensor)
            
        images_tensor = torch.stack(images) 
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)
        
        return images_tensor, kinematics_tensor, label_tensor