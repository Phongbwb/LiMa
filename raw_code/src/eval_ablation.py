import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import model ablation của bạn (Hãy chắc chắn file này chứa class LiMaVLM không có TrajectoryBranch)
from models.lima_nocfc import LiMaVLM

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
# 2. DATASET ĐÃ NÂNG CẤP (CHỈ CÓ TEMPORAL ENSEMBLING, KHÔNG TRAJECTORY)
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

        for idxs in all_segment_indices:
            crop_list, frame_list = [], []
            
            for i, f_idx in enumerate(idxs):
                frame_path = os.path.join(self.dataset_dir, track["frames"][f_idx])
                frame = default_loader(frame_path)
                
                raw_box = track["boxes"][f_idx]
                
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

        return {
            "crop": torch.stack(crop_segments), 
            "frame": torch.stack(frame_segments),
            "track_idx": index
        }


# ==========================================
# 3. HÀM ĐÁNH GIÁ TỪ FILE JSON ĐÃ XUẤT
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
            if rank <= 10:  # 🔥 BỔ SUNG TÍNH RECALL@10
                recall_10 += 1.0

    return mrr / num_queries, recall_5 / num_queries, recall_10 / num_queries


# ==========================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
    print(f"🚀 Bắt đầu Benchmark CityFlow-NL (Ablation Test Set) trên {device.upper()}...")

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
    # 🔥 Đảm bảo num_classes trùng với lúc bạn train mô hình Ablation
    model = LiMaVLM(d_model=256, d_text_in=512, num_blocks=4, num_classes=2155) 
    
    # Đường dẫn file checkpoint của mô hình Ablation
    checkpoint_path = "./checkpoints/models/ablationnewlimavlm_epoch_145.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"✅ Đã tải checkpoint từ {checkpoint_path}")
    else:
        print(f"⚠️ CẢNH BÁO: Không tìm thấy checkpoint {checkpoint_path}!")

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
    # BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG TRACKS (THUẦN VISUAL)
    # =======================================================
    print("🎬 Đang trích xuất đặc trưng Video...")
    track_embeddings_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting Tracks"):
            crops = batch["crop"].to(device)
            frames = batch["frame"].to(device)
            
            # Kích thước: [Batch, Segments, Time, Channels, H, W]
            B, S, T, C, H, W = crops.shape
            
            # Trải phẳng để xử lý song song các segments như là một batch lớn
            crops_flat = crops.view(B * S, T, C, H, W)
            frames_flat = frames.view(B * S, T, C, H, W)
            
            # 🔥 ĐÃ XÓA bbox_features. Chỉ truyền ảnh crop và ảnh gốc.
            outputs = model.encode_image(
                crop_frames=crops_flat, 
                original_frames=frames_flat
            )
            vis_embed = outputs["visual_embeds"]
            
            # Phục hồi chiều [Batch, Segment, Dim] và trung bình hóa các segments
            vis_embed = vis_embed.view(B, S, -1)
            avg_vis_embed = vis_embed.mean(dim=1)
            avg_vis_embed = F.normalize(avg_vis_embed, p=2, dim=-1)
            
            track_embeddings_list.append(avg_vis_embed)

    track_embeddings_tensor = torch.cat(track_embeddings_list, dim=0)

    # =======================================================
    # BƯỚC 3: TÍNH ĐIỂM, XUẤT FILE RANK CƠ BẢN VÀ ĐÁNH GIÁ
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
    baseline_rank_path = "./data/json/retrieval/baseline_rank_ablation.json" # 🔥 Đổi tên file để không đè file cũ
    with open(baseline_rank_path, "w", encoding='utf-8') as f:
        json.dump(baseline_rank, f, indent=4)
        
    print(f"✅ Đã xuất {baseline_rank_path}")

    # =======================================================
    # BƯỚC 4: RERANK & CHẤM ĐIỂM
    # =======================================================
    mrr_base, recall_5_base, recall_10_base = evaluate_rank_json(baseline_rank_path, query_to_gt_uuid_dict)
    
    print("\n" + "="*50)
    print("📈 KẾT QUẢ BASELINE (TRƯỚC RERANK)")
    print(f"🔹 MRR       : {mrr_base:.4f}  ({(mrr_base*100):.2f}%)")
    print(f"🔹 Recall@5  : {recall_5_base:.4f}  ({(recall_5_base*100):.2f}%)")
    print(f"🔹 Recall@10 : {recall_10_base:.4f}  ({(recall_10_base*100):.2f}%)") # 🔥 BỔ SUNG IN RECALL@10
    print("="*50)

    print("\n⏳ Vui lòng chạy script rerank.py của bạn. Sau khi có file 'final_rank.json'...")
    
    final_rank_path = './data/json/retrieval/final_rank.json'
    if os.path.exists(final_rank_path):
        # Nhận thêm recall_10_final
        mrr_final, recall_5_final, recall_10_final = evaluate_rank_json(final_rank_path, query_to_gt_uuid_dict)
        print("\n" + "="*50)
        print("🏆 KẾT QUẢ FINAL (SAU KHI RERANK)")
        print(f"🔹 MRR       : {mrr_final:.4f}  ({(mrr_final*100):.2f}%)")
        print(f"🔹 Recall@5  : {recall_5_final:.4f}  ({(recall_5_final*100):.2f}%)")
        print(f"🔹 Recall@10 : {recall_10_final:.4f}  ({(recall_10_final*100):.2f}%)") # 🔥 BỔ SUNG IN RECALL@10
        print("="*50)
    else:
        print(f"⚠️ Chưa tìm thấy '{final_rank_path}'. Hãy đảm bảo script Rerank đã chạy xong!")

if __name__ == "__main__":
    main()