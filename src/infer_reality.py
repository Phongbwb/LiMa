import os
import re
import json
import torch
import torch.nn.functional as F
import numpy as np
import spacy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm

# Import model của bạn
from models.lima import LiMaVLM

# ==========================================
# 1. CẤU HÌNH & TỪ ĐIỂN
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

class DataCfg:
    ROOT_DIR = "./"
    DATA_DIR = "./data"
    CITYFLOW_PATH = "cityflownl/data"
    CROP_AREA = 1.6666667
    
data_cfg = DataCfg()

SIZE_MAPPING = {"big": r"\b(huge|big|large|giant)\b", "medium": r"\b(midsized|midsize|medium(?!\s+(rate|speed))|semi)\b", "small": r"\b(smaller|small|mini)\b"}
COLOR_MAPPING = {"red": r"\b(dark-red|dark red|wine-red|reddish|burgundy|scarlet|rufous|res|red)\b", "silver": r"\b(dark silver|silver|sliver)\b", "grey": r"\b(dark gray|dark grey|taupe|grey|gray)\b", "blue": r"\b(dark-blue|dark blue|blue)\b", "maroon": r"\b(dark-maroon|maroon)\b", "white": r"\b(whitish|white|whit)\b", "black": r"\b(black|dark)\b", "purple": r"\b(purple)\b", "yellow": r"\b(metallic|yellow)\b", "orange": r"\b(orange)\b", "green": r"\b(green)\b", "gold": r"\b(golden|gold)\b", "brown": r"\b(brown-ish|brownish|sienna|brown|bay)\b", "tan": r"\b(tan)\b", "beige": r"\b(beige)\b", "bronze": r"\b(bronze)\b"}
TYPE_MAPPING = {"vehicle": r"\b(ford mustang|chevrolet|mercedes|vehicle|chevvy|toyota|subaru|sports|honda|chevy|audi|car|can)\b", "pickup-truck": r"\b(cargo pickup truck|cargo truck|pickup truck|pick up truck|pickuptruck|pick up|pickup)\b", "truck": r"\b(tractortrailer|semitruck|truck|flatbed|18 wheeler|wheeler)\b", "crossover": r"\b(cross over|crossover)\b", "coupe": r"\b(mini cooper|couple|coupe|coup)\b", "jeep": r"\b(cherokee|jeep)\b", "suv": r"\b(suv|spv|svu)\b", "sedan": r"\b(sedan|sede)\b", "hatchback": r"\b(hatchback|hatckback)\b", "van": r"\b(minivan|van)\b", "wagon": r"\b(wagon)\b", "mpv": r"\b(mpv)\b", "caravan": r"\b(caravan)\b", "bike": r"\b(bike)\b", "bus": r"\b(bus)\b", "taxi": r"\b(taxi)\b"}
MOTION_MAPPING = {"go straight": r"\b(drive straight down|keep straight down|run straight down|continue straight|continue forward|drving straight|proceed straight|drive solo down|drive straight|keeps straight|keep straight|head straight|move straight|straight down|continue down|drive forward|drive past|drive down|go straight|goes straight|move foward|run straight|travel down|move ahead|accelerate|run across|go across|cross straight|drive up|continue|go forward|slow down|move down|when down|keep run|run down|go down|run up|travel|enters|drive|enter|cross|run through|run|speed|switch lane|change lane|change lanes|switch lanes|go trough|lead|without stop)\b", "turn left": r"\b(turn slightly left|turn street left|make a left turn|make left turn|left turning|turn left|tuns left|left turn|take left|make left|goes left|go left)\b", "turn right": r"\b(turn slightly right|make a right turn|make right turn|make slight right|travel right|turn right|right turn|take right|take righ|make right|curve right|goes right|go right)\b", "stop": r"\b(make stop|pause|wait|stop)\b", "special": r"\b(take uturn|turn corner)\b"}

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Thiếu spaCy en_core_web_sm. Chạy: python -m spacy download en_core_web_sm")
    exit()

# ==========================================
# 2. XỬ LÝ TEXT (Từ Script NLP)
# ==========================================
def clean_original_text(text):
    text = text.lower().strip()
    text = re.sub(r'^(a|an|the)\s+', '', text)
    doc = nlp(text)
    clean_tokens = [token.lemma_ if token.pos_ in ["VERB", "AUX"] else token.text for token in doc]
    return re.sub(r'\s+([.,!?])', r'\1', " ".join(clean_tokens))

def get_first_match(text, mapping):
    best_match_label, best_match_raw, best_idx = "", "", float('inf')
    for label, pattern in mapping.items():
        for match in re.finditer(pattern, text):
            if match.start() < best_idx:
                best_idx, best_match_label, best_match_raw = match.start(), label, match.group(0)
    return best_match_label, best_match_raw

def get_type_match(text, mapping):
    matches = [(m.start(), m.end(), m.group(0), label) for label, pattern in mapping.items() for m in re.finditer(pattern, text)]
    if not matches: return "", ""
    matches.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    filtered = []
    for m in matches:
        if not filtered or m[1] > filtered[-1][1]: filtered.append(m)
    
    merged_start, merged_end, final_label = filtered[0][0], filtered[0][1], filtered[0][3]
    for i in range(1, len(filtered)):
        curr_start, curr_end, curr_label = filtered[i][0], filtered[i][1], filtered[i][3]
        gap = text[merged_end:curr_start].strip() if curr_start > merged_end else ""
        if curr_start <= merged_end or (len(gap) < 20 and not set(gap.split()).intersection({"and", "by", "with", "behind", "pass", "follow", "cross", "after"})):
            merged_end = max(merged_end, curr_end)
            if final_label == "vehicle" or (curr_label != "vehicle" and final_label == curr_label): final_label = curr_label
            elif curr_label != "vehicle": final_label = curr_label
        else: break
    return final_label, text[merged_start:merged_end]

def get_motion_match(text, mapping):
    matches = [(m.start(), m.group(0), label) for label, pattern in mapping.items() for m in re.finditer(pattern, text)]
    if not matches: return "", ""
    matches.sort(key=lambda x: x[0])
    for m in matches:
        if m[2] in {"turn left", "turn right", "go straight"}: return m[2], m[1]
    return matches[0][2], matches[0][1]

def extract_and_normalize(clean_text):
    text_lower = clean_text.lower()
    color, raw_color = get_first_match(text_lower, COLOR_MAPPING)
    v_type, raw_type = get_type_match(text_lower, TYPE_MAPPING) 
    mot, raw_motion = get_motion_match(text_lower, MOTION_MAPPING)

    remaining_text = text_lower
    for raw_match in [raw_color, raw_type, raw_motion]:
        if raw_match: remaining_text = remaining_text.replace(raw_match, "", 1)
            
    ctx_text = re.sub(r"\b(be|is|are|was|were|a|an|the|that|which|this|make|make a|at|on|in|with|of|by)\b", ' ', remaining_text)
    ctx_text = re.sub(r'[^\w\s]', ' ', ctx_text) 
    ctx_text = re.sub(r'\s+', ' ', ctx_text).strip()
    return color, v_type, mot, ctx_text

def encode_single_query(raw_query, text_encoder, tokenizer, lima_model):
    clean_text = clean_original_text(raw_query)
    color, v_type, mot, ctx = extract_and_normalize(clean_text)
    
    combined_str = f"{color or 'unknown color'} {v_type or 'unknown vehicle'} {mot or 'unknown motion'}".strip()
    inputs = tokenizer(combined_str, padding='max_length', truncation=True, max_length=32, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        clip_outputs = text_encoder(**inputs)
        raw_emb = clip_outputs.last_hidden_state.squeeze(0).mean(dim=0).unsqueeze(0)
        lima_emb = F.normalize(lima_model.encode_text(raw_emb), p=2, dim=-1)
    return lima_emb

# ==========================================
# 3. XỬ LÝ DATASET VIDEO (Từ Script Eval)
# ==========================================
def default_loader(path):
    return Image.open(path).convert('RGB')

class CityFlowTestTrackDataset(Dataset):
    def __init__(self, data_cfg, json_path, transform=None):
        self.data_cfg = data_cfg
        self.dataset_dir = os.path.join(data_cfg.DATA_DIR, data_cfg.CITYFLOW_PATH)
        with open(json_path, 'r', encoding='utf-8') as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.transform = transform
        self.num_segments = 3       
        self.frames_per_segment = 5 

    def __len__(self):
        return len(self.list_of_uuids)

    def _compute_trajectory_features(self, raw_boxes, img_w, img_h):
        T = len(raw_boxes)
        features = np.zeros((T, 18), dtype=np.float32)
        cx, cy, w, h = [], [], [], []
        for box in raw_boxes:
            cx.append((box[0] + box[2] / 2.0) / img_w) 
            cy.append((box[1] + box[3] / 2.0) / img_h)
            w.append(box[2] / img_w)
            h.append(box[3] / img_h)
        cx, cy, w, h = np.array(cx), np.array(cy), np.array(w), np.array(h)
        area, aspect_ratio = w * h, w / (h + 1e-6)

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
        sin_theta[mask], cos_theta[mask] = vy[mask] / speed[mask], vx[mask] / speed[mask]

        ax, ay = diff(vx), diff(vy)
        acc, jerk = np.sqrt(ax**2 + ay**2), diff(np.sqrt(ax**2 + ay**2))

        sin_prev, cos_prev = np.roll(sin_theta, shift=1), np.roll(cos_theta, shift=1)
        if T > 1: sin_prev[0], cos_prev[0] = sin_prev[1], cos_prev[1]
        curvature = np.arctan2(sin_theta * cos_prev - cos_theta * sin_prev, cos_theta * cos_prev + sin_theta * sin_prev)

        features[:, 0:2] = np.column_stack((cx, cy))
        features[:, 2:5] = np.column_stack((vx, vy, speed))
        features[:, 5:8] = np.column_stack((ax, ay, acc))
        features[:, 8:12] = np.column_stack((sin_theta, cos_theta, jerk, curvature))
        features[:, 12:16] = np.column_stack((w, h, area, aspect_ratio))
        features[:, 16:18] = np.column_stack((diff(area), diff(aspect_ratio)))
        return torch.tensor(features, dtype=torch.float32)

    def __getitem__(self, index):
        track = self.list_of_tracks[index]
        num_frames = len(track["frames"])
        
        all_segment_indices = []
        if num_frames >= self.num_segments * self.frames_per_segment:
            step = num_frames / self.num_segments
            for s in range(self.num_segments):
                start, end = int(s * step), int((s + 1) * step) - 1
                all_segment_indices.append([int(start + i * (end - start) / (self.frames_per_segment - 1)) for i in range(self.frames_per_segment)])
        else:
            idxs = [int(i * (num_frames - 1) / (self.frames_per_segment - 1)) for i in range(self.frames_per_segment)]
            all_segment_indices = [idxs] * self.num_segments

        crop_segments, frame_segments, bbox_segments = [], [], []

        for idxs in all_segment_indices:
            crop_list, frame_list, raw_box_list = [], [], []
            img_w, img_h = 1920, 1080
            
            for i, f_idx in enumerate(idxs):
                frame_path = os.path.join(self.dataset_dir, track["frames"][f_idx])
                frame = default_loader(frame_path)
                if i == 0: img_w, img_h = frame.size
                
                raw_box = track["boxes"][f_idx]
                raw_box_list.append(raw_box)
                
                c = self.data_cfg.CROP_AREA
                if c == 1.6666667:
                    box_expanded = (int(raw_box[0]-raw_box[2]/3.), int(raw_box[1]-raw_box[3]/3.), int(raw_box[0]+4*raw_box[2]/3.), int(raw_box[1]+4*raw_box[3]/3.))
                else:
                    box_expanded = (int(raw_box[0]-(c-1)*raw_box[2]/2.), int(raw_box[1]-(c-1)*raw_box[3]/2), int(raw_box[0]+(c+1)*raw_box[2]/2.), int(raw_box[1]+(c+1)*raw_box[3]/2.))
                
                crop = frame.crop(box_expanded).convert('RGB')
                if self.transform:
                    crop, frame = self.transform(crop), self.transform(frame)
                
                crop_list.append(crop)
                frame_list.append(frame)

            crop_segments.append(torch.stack(crop_list))
            frame_segments.append(torch.stack(frame_list))
            bbox_segments.append(self._compute_trajectory_features(raw_box_list, img_w, img_h))

        return {
            "crop": torch.stack(crop_segments), 
            "frame": torch.stack(frame_segments),
            "bbox_features": torch.stack(bbox_segments),
            "track_idx": index
        }

# ==========================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
    print(f" Khởi động Interactive Inference Li-Ma VLM trên {device.upper()}...")
    
    # 4.1 Khởi tạo Model
    print(" Đang tải Model...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.eval()
    
    model = LiMaVLM(d_model=256, d_text_in=512, num_blocks=3, num_classes=2155) 
    checkpoint_path = "./checkpoints/models/3blockslimavlm_epoch_220.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

    # 4.2 Xử lý Video Database (Smart Cache)
    track_db_path = './data/data/track_db_cached.pt'
    
    if os.path.exists(track_db_path):
        print(f" Đã tìm thấy Cache Video ({track_db_path}). Đang load thẳng vào RAM...")
        track_db = torch.load(track_db_path, map_location=device)
        track_embeddings = track_db['embeddings']
        track_uuids = track_db['uuids']
    else:
        print(" Lần chạy đầu tiên: Đang load toàn bộ video folder và trích xuất đặc trưng...")
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = CityFlowTestTrackDataset(data_cfg=data_cfg, json_path="./data/json/test-tracks.json", transform=test_transform)
        # Giữ batch_size nhỏ gọn cho RTX 4060
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
        track_uuids = test_dataset.list_of_uuids 
        track_embeddings_list = []

        with torch.inference_mode(): # Tối ưu RAM tốt hơn no_grad()
            for batch in tqdm(test_loader, desc="Extracting Videos"):
                crops = batch["crop"].to(device)
                frames = batch["frame"].to(device)
                bbox_features = batch["bbox_features"].to(device)
                
                B, S, T, C, H, W = crops.shape
                outputs = model.encode_image(
                    crop_frames=crops.view(B * S, T, C, H, W), 
                    original_frames=frames.view(B * S, T, C, H, W), 
                    bbox_features=bbox_features.view(B * S, T, 18)
                )
                vis_embed = outputs["visual_embeds"].view(B, S, -1)
                avg_vis_embed = F.normalize(vis_embed.mean(dim=1), p=2, dim=-1)
                track_embeddings_list.append(avg_vis_embed)

        track_embeddings = torch.cat(track_embeddings_list, dim=0)
        
        # Lưu lại để lần sau không phải chạy vòng lặp này nữa
        os.makedirs("./data/data", exist_ok=True)
        torch.save({'embeddings': track_embeddings, 'uuids': track_uuids}, track_db_path)
        print(f" Đã lưu cache thành công tại {track_db_path}")

    # 4.3 Vòng lặp Chat
    print("\n" + "="*60)
    print(" HỆ THỐNG TRUY VẤN VĂN BẢN (Gõ 'exit' hoặc 'quit' để thoát)")
    print("="*60)

    while True:
        user_query = input("\n Nhập câu mô tả (VD: 'A red sedan going straight'): ")
        if user_query.lower() in ['exit', 'quit']:
            break
        if not user_query.strip():
            continue
            
        query_embed = encode_single_query(user_query, text_encoder, tokenizer, model)
        
        with torch.inference_mode():
            scores = torch.matmul(query_embed, track_embeddings.T).squeeze(0)
            top_k = 5
            top_scores, top_indices = torch.topk(scores, k=top_k)
            
        print(f"\n KẾT QUẢ TÌM KIẾM CHO: '{user_query}'")
        for i in range(top_k):
            idx = top_indices[i].item()
            score = top_scores[i].item()
            print(f"  {i+1}. Track UUID: {track_uuids[idx]} | Cosine Score: {score:.4f}")

if __name__ == "__main__":
    main()