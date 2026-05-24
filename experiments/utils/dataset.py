import json
import os
import random
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torch.nn.functional as F

def default_loader(path):
    return Image.open(path).convert('RGB')

class CityFlowNLDataset(Dataset):
    def __init__(self, data_cfg, json_path, text_emb_path, transform=None, Random=True, type=None, finetune=False):
        """
        Dataset chuyên biệt cho kiến trúc Two-Stream MambaVLM.
        :param data_cfg: CfgNode chứa cấu hình đường dẫn (DATA_DIR, CITYFLOW_PATH).
        :param json_path: Đường dẫn tới file JSON chứa tracks (train/val).
        :param text_emb_path: Đường dẫn tới file .pt chứa CLIP text embeddings đã bóc tách.
        """
        self.data_cfg = data_cfg
        self.crop_area = data_cfg.CROP_AREA
        self.dataset_dir = os.path.join(self.data_cfg.DATA_DIR, self.data_cfg.CITYFLOW_PATH)
        self.json_dir = os.path.join(self.data_cfg.ROOT_DIR, json_path)
        
        self.random = Random
        self.finetune = finetune
        self.type = type
        
        # --- CẤU HÌNH SEQUENCE ---
        self.num_frames_to_sample = 4  # Số lượng frame trong chuỗi thời gian
        self.blur_radius = 10          # Độ mờ của background (Ảnh Spatial)
        
        # --- 1. TẢI TEXT EMBEDDINGS ---
        print(f"[{self.type}] Đang tải Text Embeddings từ: {text_emb_path}")
        self.text_embs = torch.load(text_emb_path, map_location="cpu")
        
        # --- 2. TẢI FILE JSON TRỢ GIÚP ---
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

    def __getitem__(self, index):
        tmp_index = self.all_indexs[index]
        track = self.list_of_tracks[tmp_index]
        
        # ==========================================
        # 1. LẤY VÀ XỬ LÝ TEXT QUERIES
        # ==========================================
        if self.random:
            nl_idx = random.randint(0, len(track["nl"]) - 1)
        else:
            nl_idx = 1 if len(track["nl"]) > 1 else 0
            
        if self.finetune:
            nl_idx = 0
            
        raw_text = track["nl"][nl_idx]
        # Chuẩn hóa text để làm key tra cứu embedding (Khớp với tool bóc tách)
        clean_text = raw_text.strip().lower()

        # ==========================================
        # 2. LẤY MẪU CHUỖI 4 FRAMES
        # ==========================================
        num_frames_in_track = len(track["frames"])
        
        if self.random:
            if num_frames_in_track >= self.num_frames_to_sample:
                frame_indices = random.sample(range(num_frames_in_track), self.num_frames_to_sample)
            else:
                frame_indices = [random.randint(0, num_frames_in_track - 1) for _ in range(self.num_frames_to_sample)]
        else:
            frame_indices = [i % num_frames_in_track for i in range(self.num_frames_to_sample)]

        crop_list = []
        frame_list = []
        blurred_frame_list = [] 

        # ==========================================
        # 3. XỬ LÝ HÌNH ẢNH (SPATIAL & CONTEXT)
        # ==========================================
        for f_idx in frame_indices:
            frame_path = os.path.join(self.dataset_dir, track["frames"][f_idx])
            frame = default_loader(frame_path)
            
            raw_box = track["boxes"][f_idx]
            exact_box = (
                int(raw_box[0]), int(raw_box[1]), 
                int(raw_box[0] + raw_box[2]), int(raw_box[1] + raw_box[3])
            )
            
            # --- LUỒNG SPATIAL 1: LÀM MỜ NỀN ---
            blurred_frame = frame.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
            sharp_car = frame.crop(exact_box)
            blurred_frame.paste(sharp_car, exact_box)
            
            # --- LUỒNG SPATIAL 2: CẮT RIÊNG BBOX ---
            if self.crop_area == 1.6666667:
                box_expanded = (int(raw_box[0]-raw_box[2]/3.), int(raw_box[1]-raw_box[3]/3.), int(raw_box[0]+4*raw_box[2]/3.), int(raw_box[1]+4*raw_box[3]/3.))
            else:
                box_expanded = (int(raw_box[0]-(self.crop_area-1)*raw_box[2]/2.), int(raw_box[1]-(self.crop_area-1)*raw_box[3]/2), int(raw_box[0]+(self.crop_area+1)*raw_box[2]/2.), int(raw_box[1]+(self.crop_area+1)*raw_box[3]/2.))
            
            crop = frame.crop(box_expanded)
            
            # ==========================================================
            #  SỬA LỖI Ở ĐÂY: Ép buộc tất cả ảnh phải là RGB (3 Kênh)
            # ==========================================================
            crop = crop.convert('RGB')
            frame = frame.convert('RGB')
            blurred_frame = blurred_frame.convert('RGB')
            
            # --- ÁP DỤNG DATA AUGMENTATION ---
            if self.transform is not None:
                crop = self.transform(crop)
                frame = self.transform(frame)
                blurred_frame = self.transform(blurred_frame)
            
            crop_list.append(crop)
            frame_list.append(frame)
            blurred_frame_list.append(blurred_frame)

        # Gộp danh sách thành Tensors 4D: [T, C, H, W]
        if self.transform is not None:
            crop_tensor = torch.stack(crop_list)             
            frame_tensor = torch.stack(frame_list)           
            blurred_frame_tensor = torch.stack(blurred_frame_list) 
        else:
            crop_tensor = crop_list
            frame_tensor = frame_list
            blurred_frame_tensor = blurred_frame_list

        # ==========================================
        # 4. ĐÓNG GÓI OUTPUT DICTIONARY
        # ==========================================
        data = {
            # HÌNH ẢNH
            "crop": crop_tensor,                   # Ảnh chỉ cắt vùng xe (Spatial Option 1)
            "blurred_frame": blurred_frame_tensor, # Ảnh toàn cảnh làm mờ nền (Spatial Option 2)
            "frame": frame_tensor,                 # Ảnh gốc toàn cảnh (Dành cho Context Branch)
            
            # METADATA
            "text": raw_text,                      
            "car_id": tmp_index,                   # ID thứ tự của xe (Dùng cho nhánh Re-ID)
            "track_uuid": self.list_of_uuids[tmp_index] 
        }

        # KẾT NỐI TEXT EMBEDDINGS TỪ FILE .PT
        if clean_text in self.text_embs:
            emb_dict = self.text_embs[clean_text]
            data.update({
                "color_embedding": emb_dict["color_embedding"],       # Shape: [8, 512]
                "type_embedding": emb_dict["type_embedding"],         # Shape: [8, 512]
                "motion_embedding": emb_dict["motion_embedding"],     # Shape: [16, 512]
                "context_embedding": emb_dict["context_embedding"],   # Shape: [16, 512]
                "color_input_ids": emb_dict["color_input_ids"],
                "type_input_ids": emb_dict["type_input_ids"],
                "motion_input_ids": emb_dict["motion_input_ids"],
                "context_input_ids": emb_dict["context_input_ids"],
                
                # ĐÃ THÊM: Embeddings của câu tổng hợp (Color + Type + Motion)
                "text_embeds": emb_dict["text_embeds"],             # Shape: [32, 512]
                "text_embeds_ids": emb_dict["text_embeds_ids"],     # Shape: [32]
                "text_embeds_text": emb_dict["text_embeds_text"],   # String
            })
        else:
            # Fallback an toàn nếu có câu text chưa được encode
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
                
                # ĐÃ THÊM: Fallback cho text_embeds (max_len = 32 khớp với script trích xuất)
                "text_embeds": torch.zeros((32, 512), dtype=torch.float32),
                "text_embeds_ids": torch.zeros((32,), dtype=torch.long),
                "text_embeds_text": "unknown vehicle",
            })

        return data