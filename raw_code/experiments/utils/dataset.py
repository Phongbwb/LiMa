import json
import os
import random
import torch
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torch.nn.functional as F

def default_loader(path):
    return Image.open(path).convert('RGB')

class CityFlowNLDataset(Dataset):
    def __init__(self, data_cfg, json_path, text_emb_path, transform=None, Random=True, type=None, finetune=False):
        self.data_cfg = data_cfg
        self.crop_area = data_cfg.CROP_AREA
        self.dataset_dir = os.path.join(self.data_cfg.DATA_DIR, self.data_cfg.CITYFLOW_PATH)
        self.json_dir = os.path.join(self.data_cfg.ROOT_DIR, json_path)
        
        self.random = Random
        self.finetune = finetune
        self.type = type
        
        # --- CẤU HÌNH SEQUENCE ---
        self.num_frames_to_sample = 5  
        self.blur_radius = 10          
        self.max_frame_stride = 3      # [MỚI] Khoảng cách tối đa (gap) giữa 2 frame liên tiếp
        
        print(f"[{self.type}] Đang tải Text Embeddings từ: {text_emb_path}")
        self.text_embs = torch.load(text_emb_path, map_location="cpu")
        
        with open(self.json_dir) as f:
            tracks = json.load(f)
            if self.type == "train":
                print(f"Loading JSON for training from: {self.json_dir}")
            else:
                print(f"Loading JSON for evaluation from: {self.json_dir}")

        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.transform = transform
        
        self.all_indexs = list(range(len(self.list_of_uuids)))
        print(f"[{self.type}] Tổng số lượng xe (tracks): {len(self.all_indexs)}")

    def __len__(self):
        return len(self.all_indexs)

    def _compute_trajectory_features(self, raw_boxes, img_w, img_h):
        # [GIỮ NGUYÊN NHƯ BẢN TRƯỚC] 
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

        sin_prev = np.roll(sin_theta, shift=1)
        cos_prev = np.roll(cos_theta, shift=1)
        if T > 1:
            sin_prev[0], cos_prev[0] = sin_prev[1], cos_prev[1]
            
        sin_diff = sin_theta * cos_prev - cos_theta * sin_prev
        cos_diff = cos_theta * cos_prev + sin_theta * sin_prev
        curvature = np.arctan2(sin_diff, cos_diff)

        area_change = diff(area)
        ratio_change = diff(aspect_ratio)

        features[:, 0], features[:, 1] = cx, cy
        features[:, 2], features[:, 3], features[:, 4] = vx, vy, speed
        features[:, 5], features[:, 6], features[:, 7] = ax, ay, acc
        features[:, 8], features[:, 9] = sin_theta, cos_theta
        features[:, 10], features[:, 11] = jerk, curvature
        features[:, 12], features[:, 13], features[:, 14], features[:, 15] = w, h, area, aspect_ratio
        features[:, 16], features[:, 17] = area_change, ratio_change

        return torch.tensor(features, dtype=torch.float32)

    def __getitem__(self, index):
        tmp_index = self.all_indexs[index]
        track = self.list_of_tracks[tmp_index]
        
        # 1. TEXT QUERIES
        if self.random:
            nl_idx = random.randint(0, len(track["nl"]) - 1)
        else:
            nl_idx = 1 if len(track["nl"]) > 1 else 0
            
        if self.finetune:
            nl_idx = 0
            
        raw_text = track["nl"][nl_idx]
        clean_text = raw_text.strip().lower()

        # ==========================================
        # 2. LẤY MẪU CHUỖI 5 FRAMES (THEO THỨ TỰ & KHOẢNG CÁCH ĐỀU)
        # ==========================================
        num_frames_in_track = len(track["frames"])
        
        if self.random:
            if num_frames_in_track >= self.num_frames_to_sample:
                # Tìm bước nhảy (stride) tối đa có thể với số lượng frame hiện tại
                possible_max_stride = (num_frames_in_track - 1) // (self.num_frames_to_sample - 1)
                
                # Chốt khoảng cách thực tế (không vượt quá ngưỡng tự định nghĩa)
                actual_max_stride = min(self.max_frame_stride, possible_max_stride)
                actual_max_stride = max(1, actual_max_stride) # Đảm bảo tối thiểu là 1
                
                # Lấy ngẫu nhiên khoảng cách (gap) để mô hình học tốc độ di chuyển đa dạng
                stride = random.randint(1, actual_max_stride)
                
                # Chiều dài của chuỗi con
                seq_length = (self.num_frames_to_sample - 1) * stride + 1
                
                # Chọn frame bắt đầu ngẫu nhiên
                max_start_idx = num_frames_in_track - seq_length
                start_idx = random.randint(0, max_start_idx)
                
                # Tạo list index (luôn đúng thứ tự và cách đều nhau)
                frame_indices = [start_idx + i * stride for i in range(self.num_frames_to_sample)]
            else:
                # Quá ngắn: đành phải lặp lại frame nhưng ép phải sort để giữ thứ tự thời gian
                frame_indices = sorted([random.randint(0, num_frames_in_track - 1) for _ in range(self.num_frames_to_sample)])
        else:
            # Chế độ Val/Test: Lấy frame căn giữa với khoảng cách tối ưu
            if num_frames_in_track >= self.num_frames_to_sample:
                possible_max_stride = (num_frames_in_track - 1) // (self.num_frames_to_sample - 1)
                stride = min(self.max_frame_stride, possible_max_stride)
                stride = max(1, stride)
                
                seq_length = (self.num_frames_to_sample - 1) * stride + 1
                start_idx = (num_frames_in_track - seq_length) // 2  # Căn giữa đoạn video
                
                frame_indices = [start_idx + i * stride for i in range(self.num_frames_to_sample)]
            else:
                # Quá ngắn: Phân bố đều
                frame_indices = sorted([i % num_frames_in_track for i in range(self.num_frames_to_sample)])

        # ==========================================
        # 3. XỬ LÝ HÌNH ẢNH & TỌA ĐỘ
        # ==========================================
        crop_list, frame_list, blurred_frame_list = [], [], []
        raw_box_list = [] 
        img_w, img_h = 1920, 1080 

        for idx, f_idx in enumerate(frame_indices):
            frame_path = os.path.join(self.dataset_dir, track["frames"][f_idx])
            frame = default_loader(frame_path)
            
            if idx == 0:
                img_w, img_h = frame.size
            
            raw_box = track["boxes"][f_idx]
            raw_box_list.append(raw_box) 
            
            exact_box = (
                int(raw_box[0]), int(raw_box[1]), 
                int(raw_box[0] + raw_box[2]), int(raw_box[1] + raw_box[3])
            )
            
            blurred_frame = frame.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
            sharp_car = frame.crop(exact_box)
            blurred_frame.paste(sharp_car, exact_box)
            
            if self.crop_area == 1.6666667:
                box_expanded = (int(raw_box[0]-raw_box[2]/3.), int(raw_box[1]-raw_box[3]/3.), int(raw_box[0]+4*raw_box[2]/3.), int(raw_box[1]+4*raw_box[3]/3.))
            else:
                box_expanded = (int(raw_box[0]-(self.crop_area-1)*raw_box[2]/2.), int(raw_box[1]-(self.crop_area-1)*raw_box[3]/2), int(raw_box[0]+(self.crop_area+1)*raw_box[2]/2.), int(raw_box[1]+(self.crop_area+1)*raw_box[3]/2.))
            
            crop = frame.crop(box_expanded)
            
            crop = crop.convert('RGB')
            frame = frame.convert('RGB')
            blurred_frame = blurred_frame.convert('RGB')
            
            if self.transform is not None:
                crop = self.transform(crop)
                frame = self.transform(frame)
                blurred_frame = self.transform(blurred_frame)
            
            crop_list.append(crop)
            frame_list.append(frame)
            blurred_frame_list.append(blurred_frame)

        if self.transform is not None:
            crop_tensor = torch.stack(crop_list)             
            frame_tensor = torch.stack(frame_list)           
            blurred_frame_tensor = torch.stack(blurred_frame_list) 
        else:
            crop_tensor, frame_tensor, blurred_frame_tensor = crop_list, frame_list, blurred_frame_list

        bbox_features_tensor = self._compute_trajectory_features(raw_box_list, img_w, img_h)

        # 4. ĐÓNG GÓI OUTPUT
        data = {
            "crop": crop_tensor,                   
            "blurred_frame": blurred_frame_tensor, 
            "frame": frame_tensor,                 
            "bbox_features": bbox_features_tensor, 
            "text": raw_text,                      
            "car_id": tmp_index,                   
            "track_uuid": self.list_of_uuids[tmp_index] 
        }

        if clean_text in self.text_embs:
            emb_dict = self.text_embs[clean_text]
            data.update({
                "color_embedding": emb_dict["color_embedding"],       
                "type_embedding": emb_dict["type_embedding"],         
                "motion_embedding": emb_dict["motion_embedding"],     
                "context_embedding": emb_dict["context_embedding"],   
                "color_input_ids": emb_dict["color_input_ids"],
                "type_input_ids": emb_dict["type_input_ids"],
                "motion_input_ids": emb_dict["motion_input_ids"],
                "context_input_ids": emb_dict["context_input_ids"],
                "text_embeds": emb_dict["text_embeds"],             
                "text_embeds_ids": emb_dict["text_embeds_ids"],     
                "text_embeds_text": emb_dict["text_embeds_text"],   
            })
        else:
            print(f"[Warning] Text embedding not found for: '{clean_text}'")
            data.update({
                "color_embedding": torch.zeros((8, 512), dtype=torch.float32),
                "type_embedding": torch.zeros((8, 512), dtype=torch.float32),
                "motion_embedding": torch.zeros((16, 512), dtype=torch.float32),
                "context_embedding": torch.zeros((16, 512), dtype=torch.float32),
                "color_input_ids": torch.zeros((8,), dtype=torch.long),
                "type_input_ids": torch.zeros((8,), dtype=torch.long),
                "motion_input_ids": torch.zeros((16,), dtype=torch.long),
                "context_input_ids": torch.zeros((16,), dtype=torch.long),
                "text_embeds": torch.zeros((32, 512), dtype=torch.float32),
                "text_embeds_ids": torch.zeros((32,), dtype=torch.long),
                "text_embeds_text": "unknown vehicle",
            })

        return data