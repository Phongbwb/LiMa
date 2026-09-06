import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import model của bạn
from models.lima import LiMaVLM

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
class DataCfg:
    ROOT_DIR = "./"
    DATA_DIR = "./data"
    CITYFLOW_PATH = "cityflownl/data"
    CROP_AREA = 1.6666667
    
data_cfg = DataCfg()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. DATASET ĐÃ NÂNG CẤP (TEMPORAL ENSEMBLING + TRAJECTORY)
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
        
        # Cấu hình Temporal Ensembling
        self.num_segments = 3       
        self.frames_per_segment = 5 

    def __len__(self):
        return len(self.list_of_uuids)

    # Hàm tính Trajectory Features
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
        track = self.list_of_tracks[index]
        num_frames = len(track["frames"])
        
        all_segment_indices = []
        if num_frames >= self.num_segments * self.frames_per_segment:
            step = num_frames / self.num_segments
            for s in range(self.num_segments):
                start = int(s * step)
                end = int((s + 1) * step) - 1
                idxs = [int(start + i * (end - start) / (self.frames_per_segment - 1)) for i in range(self.frames_per_segment)]
                all_segment_indices.append(idxs)
        else:
            idxs = [int(i * (num_frames - 1) / (self.frames_per_segment - 1)) for i in range(self.frames_per_segment)]
            all_segment_indices = [idxs] * self.num_segments

        crop_segments = []
        frame_segments = []
        bbox_segments = []

        for idxs in all_segment_indices:
            crop_list, frame_list, raw_box_list = [], [], []
            img_w, img_h = 1920, 1080
            
            for i, f_idx in enumerate(idxs):
                frame_path = os.path.join(self.dataset_dir, track["frames"][f_idx])
                frame = default_loader(frame_path)
                
                if i == 0:
                    img_w, img_h = frame.size
                
                raw_box = track["boxes"][f_idx]
                raw_box_list.append(raw_box) 
                
                if self.data_cfg.CROP_AREA == 1.6666667:
                    box_expanded = (int(raw_box[0]-raw_box[2]/3.), int(raw_box[1]-raw_box[3]/3.), int(raw_box[0]+4*raw_box[2]/3.), int(raw_box[1]+4*raw_box[3]/3.))
                else:
                    c = self.data_cfg.CROP_AREA
                    box_expanded = (int(raw_box[0]-(c-1)*raw_box[2]/2.), int(raw_box[1]-(c-1)*raw_box[3]/2), int(raw_box[0]+(c+1)*raw_box[2]/2.), int(raw_box[1]+(c+1)*raw_box[3]/2.))
                
                crop = frame.crop(box_expanded).convert('RGB')
                
                if self.transform is not None:
                    crop = self.transform(crop)
                    frame = self.transform(frame)
                
                crop_list.append(crop)
                frame_list.append(frame)

            crop_segments.append(torch.stack(crop_list))
            frame_segments.append(torch.stack(frame_list))
            
            bbox_features = self._compute_trajectory_features(raw_box_list, img_w, img_h)
            bbox_segments.append(bbox_features)

        return {
            "crop": torch.stack(crop_segments), 
            "frame": torch.stack(frame_segments),
            "bbox_features": torch.stack(bbox_segments), 
            "track_idx": index
        }


# ==========================================
# 3. CÁC HÀM ĐÁNH GIÁ & PHÂN TÍCH LỖI
# ==========================================
def evaluate_rank_json(rank_json_path, gt_mapping_dict):
    """Tính MRR, Recall@5 và Recall@10"""
    with open(rank_json_path, 'r', encoding='utf-8') as f:
        rank_data = json.load(f)

    mrr, recall_5, recall_10 = 0.0, 0.0, 0.0
    num_queries = len(rank_data)

    for query_uuid, ranked_track_uuids in rank_data.items():
        gt_track_uuid = gt_mapping_dict.get(query_uuid)
        
        if gt_track_uuid in ranked_track_uuids:
            rank = ranked_track_uuids.index(gt_track_uuid) + 1
            mrr += 1.0 / rank
            
            if rank <= 5:
                recall_5 += 1.0
            if rank <= 10:  
                recall_10 += 1.0

    return mrr / num_queries, recall_5 / num_queries, recall_10 / num_queries

def export_failure_cases(rank_dict, gt_mapping_dict, queries_data, sim_matrix, query_uuids_list, list_of_track_uuids, save_path, top_k_threshold=5):
    """Tìm và xuất các query mà Ground Truth bị xếp hạng > top_k_threshold."""
    print(f"\n🔍 Đang phân tích định tính các trường hợp thất bại (GT Rank > {top_k_threshold})...")
    failure_cases = []
    
    # Tạo dict mapping track_uuid -> index để lấy điểm cosine similarity nhanh (O(1))
    track_uuid_to_idx = {uuid: idx for idx, uuid in enumerate(list_of_track_uuids)}

    for i, query_uuid in enumerate(query_uuids_list):
        ranked_tracks = rank_dict[query_uuid]
        gt_track = gt_mapping_dict.get(query_uuid)
        
        if gt_track not in ranked_tracks:
            continue
            
        gt_rank = ranked_tracks.index(gt_track) + 1
        
        # Nếu mô hình dự đoán trật khỏi Top-K
        if gt_rank > top_k_threshold:
            q_info = queries_data.get(query_uuid, {})
            scores = sim_matrix[i] 
            
            gt_score = float(scores[track_uuid_to_idx[gt_track]].item())
            
            # Lấy thông tin các tracks chen chân vào Top K
            top_k_predictions = []
            for track_id in ranked_tracks[:top_k_threshold]:
                t_score = float(scores[track_uuid_to_idx[track_id]].item())
                top_k_predictions.append({
                    "track_uuid": track_id,
                    "score": round(t_score, 4)
                })

            failure_cases.append({
                "query_uuid": query_uuid,
                "text_query": {
                    "nl": q_info.get("nl", [""])[0],
                    "color": q_info.get("color", [""])[0],
                    "type": q_info.get("type", [""])[0],
                    "motion": q_info.get("motion", [""])[0]
                },
                "ground_truth": {
                    "track_uuid": gt_track,
                    "rank": gt_rank,
                    "score": round(gt_score, 4)
                },
                f"top_{top_k_threshold}_false_positives": top_k_predictions
            })

    # Lưu ra file JSON
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(failure_cases, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Đã tìm thấy {len(failure_cases)} trường hợp thất bại. Lưu tại: {save_path}")

# ==========================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
    print(f"🚀 Bắt đầu Benchmark CityFlow-NL (Test Set) trên {device.upper()}...")

    # --- 4.1 Chuẩn bị Dữ liệu ---
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = CityFlowTestTrackDataset(
        data_cfg=data_cfg,
        json_path="./data/json/test-tracks.json",
        transform=test_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    list_of_track_uuids = test_dataset.list_of_uuids 
    
    # --- 4.2 Khởi tạo Model ---
    print("🧠 Tải trọng số mô hình...")
    model = LiMaVLM(d_model=256, d_text_in=512, num_blocks=4, num_classes=2155) 
    checkpoint = torch.load("./checkpoints/models/newlimavlm_epoch_210.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=False) 
    model.to(device)
    model.eval()

    # =======================================================
    # BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG VĂN BẢN
    # =======================================================
    print("📝 Đang trích xuất đặc trưng Văn bản...")
    with open("./data/json/test-queries-clean.json", 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
        
    text_embs_dict = torch.load("./data/data/clip_text_tokens_extracted.pt", map_location="cpu")
    
    query_embeddings_list = []
    query_uuids_list = []
    query_to_gt_uuid_dict = {} 

    with torch.no_grad():
        for track_idx, (query_uuid, q_data) in enumerate(queries_data.items()):
            if track_idx >= len(test_dataset.list_of_uuids): break
                
            def get_and_encode(text_key):
                texts = q_data.get(text_key, [])
                if not texts: return None
                clean_text = texts[0].strip().lower()
                if clean_text in text_embs_dict:
                    raw_emb = text_embs_dict[clean_text]["text_embeds"].to(device)
                    if raw_emb.dim() == 2: raw_emb = raw_emb.mean(dim=0).unsqueeze(0) 
                    return model.encode_text(raw_emb)
                return None

            proj_nl = get_and_encode("nl")
            proj_color = get_and_encode("color")
            proj_type = get_and_encode("type")
            proj_motion = get_and_encode("motion")
            
            valid_projs = [p for p in [proj_nl, proj_color, proj_type, proj_motion] if p is not None]
            
            if len(valid_projs) > 0:
                avg_lang_embed = sum(valid_projs) / len(valid_projs)
                avg_lang_embed = F.normalize(avg_lang_embed, p=2, dim=-1)
            else:
                avg_lang_embed = torch.zeros(1, 256, device=device)

            query_embeddings_list.append(avg_lang_embed)
            
            query_uuids_list.append(query_uuid)
            gt_track_uuid = test_dataset.list_of_uuids[track_idx]
            query_to_gt_uuid_dict[query_uuid] = gt_track_uuid

    query_embeddings_tensor = torch.cat(query_embeddings_list, dim=0)

    # =======================================================
    # BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG TRACKS CÓ TRAJECTORY
    # =======================================================
    print("🎬 Đang trích xuất đặc trưng Video + Quỹ đạo...")
    track_embeddings_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting Tracks"):
            crops = batch["crop"].to(device)
            frames = batch["frame"].to(device)
            bbox_features = batch["bbox_features"].to(device) 
            
            B, S, T, C, H, W = crops.shape
            
            crops_flat = crops.view(B * S, T, C, H, W)
            frames_flat = frames.view(B * S, T, C, H, W)
            bbox_flat = bbox_features.view(B * S, T, 18) 
            
            outputs = model.encode_image(
                crop_frames=crops_flat, 
                original_frames=frames_flat, 
                bbox_features=bbox_flat
            )
            vis_embed = outputs["visual_embeds"]
            
            vis_embed = vis_embed.view(B, S, -1)
            avg_vis_embed = vis_embed.mean(dim=1)
            avg_vis_embed = F.normalize(avg_vis_embed, p=2, dim=-1)
            
            track_embeddings_list.append(avg_vis_embed)

    track_embeddings_tensor = torch.cat(track_embeddings_list, dim=0)

    # =======================================================
    # BƯỚC 3: TÍNH ĐIỂM VÀ TẠO BASELINE RANK
    # =======================================================
    print("\n🧮 Tính ma trận Cosine Similarity...")
    sim_matrix = torch.matmul(query_embeddings_tensor, track_embeddings_tensor.T)
    
    print("💾 Đang tạo Baseline Rank file...")
    baseline_rank = {}
    num_queries = sim_matrix.shape[0]

    for i in range(num_queries):
        query_uuid = query_uuids_list[i]
        scores = sim_matrix[i]
        
        ranked_indices = torch.argsort(scores, descending=True).cpu().numpy()
        ranked_track_uuids = [list_of_track_uuids[idx] for idx in ranked_indices]
        baseline_rank[query_uuid] = ranked_track_uuids

    os.makedirs("./data/json/retrieval", exist_ok=True)
    baseline_rank_path = "./data/json/retrieval/baseline_rank.json"
    with open(baseline_rank_path, "w", encoding='utf-8') as f:
        json.dump(baseline_rank, f, indent=4)
        
    print(f"✅ Đã xuất {baseline_rank_path}")

    # =======================================================
    # BƯỚC 3.5: PHÂN TÍCH VÀ XUẤT CÁC CASE THẤT BẠI
    # =======================================================
    failure_cases_path = "./data/json/retrieval/baseline_failure_cases.json"
    export_failure_cases(
        rank_dict=baseline_rank,
        gt_mapping_dict=query_to_gt_uuid_dict,
        queries_data=queries_data,
        sim_matrix=sim_matrix,
        query_uuids_list=query_uuids_list,
        list_of_track_uuids=list_of_track_uuids,
        save_path=failure_cases_path,
        top_k_threshold=5 # Đánh giá trượt Top 5 là lỗi
    )

    # =======================================================
    # BƯỚC 4: RERANK & CHẤM ĐIỂM
    # =======================================================
    mrr_base, recall_5_base, recall_10_base = evaluate_rank_json(baseline_rank_path, query_to_gt_uuid_dict)
    
    print("\n" + "="*50)
    print("📈 KẾT QUẢ BASELINE (TRƯỚC RERANK)")
    print(f"🔹 MRR       : {mrr_base:.4f}  ({(mrr_base*100):.2f}%)")
    print(f"🔹 Recall@5  : {recall_5_base:.4f}  ({(recall_5_base*100):.2f}%)")
    print(f"🔹 Recall@10 : {recall_10_base:.4f}  ({(recall_10_base*100):.2f}%)") 
    print("="*50)

    print("\n⏳ Vui lòng chạy script rerank.py của bạn. Sau khi có file 'final_rank.json'...")
    
    final_rank_path = './data/json/retrieval/final_rank.json'
    if os.path.exists(final_rank_path):
        mrr_final, recall_5_final, recall_10_final = evaluate_rank_json(final_rank_path, query_to_gt_uuid_dict)
        print("\n" + "="*50)
        print("🏆 KẾT QUẢ FINAL (SAU KHI RERANK)")
        print(f"🔹 MRR       : {mrr_final:.4f}  ({(mrr_final*100):.2f}%)")
        print(f"🔹 Recall@5  : {recall_5_final:.4f}  ({(recall_5_final*100):.2f}%)")
        print(f"🔹 Recall@10 : {recall_10_final:.4f}  ({(recall_10_final*100):.2f}%)") 
        print("="*50)

        # 🔥 BỔ SUNG: XUẤT FILE LỖI SAU RERANK
        print("\n🔍 Đang tải kết quả Final Rank để phân tích lỗi...")
        with open(final_rank_path, 'r', encoding='utf-8') as f:
            final_rank_dict = json.load(f)

        final_failure_cases_path = "./data/json/retrieval/final_failure_cases.json"
        export_failure_cases(
            rank_dict=final_rank_dict,
            gt_mapping_dict=query_to_gt_uuid_dict,
            queries_data=queries_data,
            sim_matrix=sim_matrix,
            query_uuids_list=query_uuids_list,
            list_of_track_uuids=list_of_track_uuids,
            save_path=final_failure_cases_path,
            top_k_threshold=5
        )
    else:
        print(f"⚠️ Chưa tìm thấy '{final_rank_path}'. Hãy đảm bảo script Rerank đã chạy xong!")

if __name__ == "__main__":
    main()