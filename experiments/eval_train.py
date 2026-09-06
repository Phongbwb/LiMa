import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.lima import LiMaVLM
from models.microlocal import MicroLocalization

# ==========================================================
# 1. HÀM TÍNH IOU
# ==========================================================
def calculate_iou_batch(pred_boxes_cxcywh, gt_boxes_tlwh):
    pred_x1 = pred_boxes_cxcywh[:, 0] - pred_boxes_cxcywh[:, 2] / 2
    pred_y1 = pred_boxes_cxcywh[:, 1] - pred_boxes_cxcywh[:, 3] / 2
    pred_x2 = pred_boxes_cxcywh[:, 0] + pred_boxes_cxcywh[:, 2] / 2
    pred_y2 = pred_boxes_cxcywh[:, 1] + pred_boxes_cxcywh[:, 3] / 2

    gt_x1 = gt_boxes_tlwh[:, 0]
    gt_y1 = gt_boxes_tlwh[:, 1]
    gt_x2 = gt_boxes_tlwh[:, 0] + gt_boxes_tlwh[:, 2]
    gt_y2 = gt_boxes_tlwh[:, 1] + gt_boxes_tlwh[:, 3]

    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    union_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1) + (gt_x2 - gt_x1) * (gt_y2 - gt_y1) - inter_area + 1e-6
    return inter_area / union_area

# ==========================================================
# 2. PIPELINE SUY LUẬN
# ==========================================================
class VideoPartGroundingPipeline:
    def __init__(self, lima_ckpt_path, micro_ckpt_path, device='cuda'):
        self.device = device
        self.lima_downsample_ratio = 16 
        
        print("Đang tải LiMaVLM...")
        self.lima_model = LiMaVLM(d_model=256, d_text=512, num_blocks=4).to(self.device)
        self.lima_model.load_state_dict(torch.load(lima_ckpt_path, map_location=self.device).get('model_state_dict', torch.load(lima_ckpt_path, map_location=self.device)))
        self.lima_model.eval()

        print("Đang tải MicroLocalization...")
        self.micro_model = MicroLocalization(embed_dim=512, num_parts=5, roi_size=7).to(self.device)
        self.micro_model.load_state_dict(torch.load(micro_ckpt_path, map_location=self.device).get('model_state_dict', torch.load(micro_ckpt_path, map_location=self.device)))
        self.micro_model.eval()

        self.roi_resizer = Resize((336, 336), antialias=True)

    def _crop_and_resize(self, orig_frames_list, bboxes_cxcywh, orig_size):
        B, T, _ = bboxes_cxcywh.shape
        W_orig, H_orig = orig_size
        scale_w_to_orig = W_orig / 384.0
        scale_h_to_orig = H_orig / 384.0
        
        cropped_rois = []
        frame_idx = 0
        for b in range(B):
            for t in range(T):
                orig_img = orig_frames_list[frame_idx] 
                frame_idx += 1
                cx_384, cy_384, w_384, h_384 = bboxes_cxcywh[b, t] * self.lima_downsample_ratio
                
                cx_orig, cy_orig = cx_384 * scale_w_to_orig, cy_384 * scale_h_to_orig
                w_orig, h_orig = w_384 * scale_w_to_orig, h_384 * scale_h_to_orig
                
                x1 = int(max(0, cx_orig - w_orig / 2))
                y1 = int(max(0, cy_orig - h_orig / 2))
                x2 = int(min(W_orig, cx_orig + w_orig / 2))
                y2 = int(min(H_orig, cy_orig + h_orig / 2))
                
                if x2 <= x1: x2 = x1 + 10
                if y2 <= y1: y2 = y1 + 10
                
                roi_pil = orig_img.crop((x1, y1, x2, y2))
                cropped_rois.append(self.roi_resizer(transforms.ToTensor()(roi_pil)))
        return torch.stack(cropped_rois).to(self.device)

    @torch.no_grad()
    def run_inference(self, video_frames, orig_frames_list, orig_size, color_text, type_text, motion_text, context_text, part_text):
        lima_output = self.lima_model(video_frames=video_frames, color_text_tokens=color_text, type_text_tokens=type_text, motion_text_tokens=motion_text, context_text_tokens=context_text)
        bboxes = lima_output["final_bboxes"]
        frame_scores = lima_output["frame_scores"]

        vehicle_rois = self._crop_and_resize(orig_frames_list, bboxes, orig_size)
        B, T = bboxes.shape[0], bboxes.shape[1]
        
        part_text_expanded = part_text.unsqueeze(1).expand(B, T, part_text.shape[1], part_text.shape[2]).reshape(B * T, part_text.shape[1], part_text.shape[2]) if part_text.shape[0] == B else part_text
        _, _, _, _, roi_match_scores = self.micro_model(vehicle_roi=vehicle_rois, word_tokens=part_text_expanded)

        final_score = (1.0 * frame_scores.max(dim=1)[0]) + (0.0 * lima_output["match_score"]) + (0.0 * roi_match_scores.squeeze(-1).view(B, T, -1).mean(dim=2).max(dim=1)[0])

        return {"final_score": final_score, "lima_bboxes": bboxes, "frame_scores": frame_scores}

# ==========================================================
# 3. BENCHMARK DÀNH RIÊNG CHO TẬP TRAIN
# ==========================================================
class TrainSetBenchmark:
    def __init__(self, pipeline, train_tracks_path, text_emb_path, data_root, num_sampled_frames=8, vis_dir="./visualizations_train", num_vis=5, max_samples=20): 
        self.pipeline = pipeline
        self.num_sampled_frames = num_sampled_frames
        self.device = pipeline.device
        self.data_root = data_root
        
        self.vis_dir = vis_dir
        self.num_vis = num_vis
        self.max_samples = max_samples
        os.makedirs(self.vis_dir, exist_ok=True)
        
        print(" Đang tải dữ liệu JSON Train...")
        with open(train_tracks_path, 'r') as f:
            all_tracks = json.load(f)
            
        print(" Đang tải Text Embeddings...")
        self.text_features_dict = torch.load(text_emb_path, map_location=self.device)
        
        # Tiền xử lý: Rút trích các cặp (Query, GT_Track) từ cấu trúc JSON lồng nhau
        self.eval_pairs = []
        track_keys = list(all_tracks.keys())[:self.max_samples] # Giới hạn số lượng track
        self.tracks_data = {k: all_tracks[k] for k in track_keys}
        
        for track_id, track_info in self.tracks_data.items():
            # Lấy câu lệnh `nl` đầu tiên làm query đại diện
            if "nl" in track_info and len(track_info["nl"]) > 0:
                query_text = track_info["nl"][0].strip().lower()
                self.eval_pairs.append({
                    "query": query_text,
                    "gt_track_id": track_id
                })
        
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize((384, 384), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.cached_videos = {}

    def preload_videos(self):
        print(f" Đang tiền xử lý {len(self.tracks_data)} Video Tracks (Train Set)...")
        for track_uuid, track_info in tqdm(self.tracks_data.items(), desc="Loading Videos"):
            total_frames = len(track_info['frames'])
            indices = np.linspace(0, total_frames - 1, self.num_sampled_frames, dtype=int)
            
            sampled_frames, sampled_boxes, orig_frames_list, orig_size = [], [], [], None      
            for idx in indices:
                img_path = track_info['frames'][idx] 
                if img_path.startswith('./'): img_path = img_path[2:]
                img_path = os.path.join(self.data_root, img_path) 
                
                img = Image.open(img_path).convert("RGB")
                if orig_size is None: orig_size = img.size 
                    
                scale_w, scale_h = 384.0 / orig_size[0], 384.0 / orig_size[1]
                orig_frames_list.append(img.copy())
                sampled_frames.append(self.transform(img))
                
                orig_box = track_info['boxes'][idx]
                sampled_boxes.append([orig_box[0]*scale_w, orig_box[1]*scale_h, orig_box[2]*scale_w, orig_box[3]*scale_h])
                
            self.cached_videos[track_uuid] = (
                torch.stack(sampled_frames, dim=1).unsqueeze(0).to(self.device), 
                torch.tensor(sampled_boxes, dtype=torch.float32).to(self.device), 
                orig_frames_list, orig_size
            )

    def visualize_predictions(self, query_text, orig_frames_list, orig_size, gt_boxes, pred_boxes, save_path, iou_score, pred_scores=None):
        T, (orig_w, orig_h) = len(orig_frames_list), orig_size
        scale_x, scale_y = orig_w / 384.0, orig_h / 384.0
        fig, axes = plt.subplots(1, T, figsize=(4 * T, 4))
        if T == 1: axes = [axes]
        
        fig.suptitle(f"TRAIN SET | Query: {query_text} | mIoU: {iou_score:.4f}\nGreen: Ground Truth, Red: Prediction", fontsize=14, fontweight='bold')

        for t in range(T):
            axes[t].imshow(orig_frames_list[t]); axes[t].axis('off')
            
            # GT Box
            gx, gy, gw, gh = gt_boxes[t].numpy()
            gx, gy, gw, gh = gx * scale_x, gy * scale_y, gw * scale_x, gh * scale_y
            axes[t].add_patch(patches.Rectangle((gx, gy), gw, gh, linewidth=2, edgecolor='lime', facecolor='none'))

            # Pred Box
            px, py, pw, ph = pred_boxes[t].numpy()
            px, py, pw, ph = px * scale_x, py * scale_y, pw * scale_x, ph * scale_y
            px1, py1 = px - pw / 2, py - ph / 2 
            axes[t].add_patch(patches.Rectangle((px1, py1), pw, ph, linewidth=2, edgecolor='red', facecolor='none', linestyle='--'))
            
            # Text Score
            if pred_scores is not None:
                score_val = pred_scores[t].item() if torch.is_tensor(pred_scores[t]) else pred_scores[t]
                axes[t].text(px1, py1 - 5, f"{score_val:.2f}", color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=1))

        plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=150); plt.close(fig)

    def evaluate(self):
        self.preload_videos()
        
        num_pairs = len(self.eval_pairs)
        track_keys = list(self.tracks_data.keys())
        num_tracks = len(track_keys)
        
        score_matrix = np.zeros((num_pairs, num_tracks))
        ranks = np.zeros(num_pairs)
        all_ious = []
        missing_embeddings = 0
        
        self.pipeline.lima_model.eval()
        self.pipeline.micro_model.eval()

        print(f" Bắt đầu Benchmark Tập TRAIN: Đánh giá {num_pairs} cặp...")
        
        with torch.no_grad():
            for i, pair in enumerate(tqdm(self.eval_pairs, desc="Đánh giá Queries")):
                query_text = pair["query"]
                gt_track_id = pair["gt_track_id"]
                
                text_data = self.text_features_dict.get(query_text, None)
                if text_data is None:
                    ranks[i] = num_tracks 
                    missing_embeddings += 1
                    continue
                    
                color_txt = text_data["color_embedding"].unsqueeze(0).to(self.device)
                type_txt = text_data["type_embedding"].unsqueeze(0).to(self.device)
                motion_txt = text_data["motion_embedding"].unsqueeze(0).to(self.device)
                context_txt = text_data["context_embedding"].unsqueeze(0).to(self.device)
                part_txt = text_data["vehicle_motion_embedding"].unsqueeze(0).to(self.device)
                
                for j, t_key in enumerate(track_keys):
                    video_tensor, gt_boxes, orig_frames_list, orig_size = self.cached_videos[t_key]
                    
                    results = self.pipeline.run_inference(
                        video_frames=video_tensor, orig_frames_list=orig_frames_list, 
                        orig_size=orig_size, color_text=color_txt, type_text=type_txt,
                        motion_text=motion_txt, context_text=context_txt, part_text=part_txt
                    )
                    score_matrix[i, j] = results["final_score"].item()
                    
                    # Nếu Track đang xét chính là Ground Truth của Query này
                    if t_key == gt_track_id: 
                        pred_bboxes = results["lima_bboxes"].squeeze(0) * self.pipeline.lima_downsample_ratio
                        iou = calculate_iou_batch(pred_bboxes, gt_boxes)
                        mean_iou_val = iou.mean().item()
                        all_ious.append(mean_iou_val)

                        if i < self.num_vis:
                            save_path = os.path.join(self.vis_dir, f"train_vis_{i:03d}.png")
                            self.visualize_predictions(
                                query_text=query_text, orig_frames_list=orig_frames_list, 
                                orig_size=orig_size, gt_boxes=gt_boxes.cpu(), 
                                pred_boxes=pred_bboxes.cpu(), save_path=save_path,
                                iou_score=mean_iou_val, pred_scores=results["frame_scores"].squeeze(0).cpu()
                            )

                # Xếp hạng
                sorted_indices = np.argsort(-score_matrix[i, :])
                gt_index_in_tracks = track_keys.index(gt_track_id)
                rank = np.where(sorted_indices == gt_index_in_tracks)[0][0] + 1 
                ranks[i] = rank

        valid_pairs = num_pairs - missing_embeddings
        if valid_pairs == 0:
            print("LỖI: Không tìm thấy Text Embeddings nào khớp với tập Train. Bạn đã extract CLIP features cho tập Train chưa?")
            return

        r1 = np.sum(ranks == 1) / valid_pairs * 100
        r5 = np.sum(ranks <= 5) / valid_pairs * 100
        mrr = np.sum(1.0 / ranks) / valid_pairs
        mIoU = np.mean(all_ious) if len(all_ious) > 0 else 0.0

        print("\n" + "="*50)
        print("🏆 KẾT QUẢ CITYFLOW-NL (TẬP TRAIN - SANITY CHECK) 🏆")
        print("="*50)
        print(f"🔹 Số câu query đánh giá    : {valid_pairs} (Thiếu embeddings: {missing_embeddings})")
        print("-" * 50)
        print(f"🔹 Mean Reciprocal Rank (MRR) : {mrr:.4f}")
        print(f"🔹 Retrieval Recall@1         : {r1:.2f}%")
        print(f"🔹 Retrieval Recall@5         : {r5:.2f}%")
        print(f"🔹 Localization mIoU          : {mIoU:.4f}")
        print(f"📷 Đã lưu {self.num_vis} ảnh phân tích Train tại: {self.vis_dir}")
        print("="*50)

# ==========================================================
# 4. CHẠY BENCHMARK
# ==========================================================
if __name__ == "__main__":
    my_pipeline = VideoPartGroundingPipeline(
        lima_ckpt_path="./data/data/checkpoint_stage1_epoch_40.pth",
        micro_ckpt_path="./checkpoints/micro_localization/best_model_epoch_59.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    benchmark = TrainSetBenchmark(
        pipeline=my_pipeline, 
        train_tracks_path="./data/data/train-tracks.json", # <--- Đường dẫn file JSON Train
        text_emb_path="./data/data/clip_text_tokens_extracted_optimized.pt", # <--- CẢNH BÁO: Đảm bảo bạn dùng file PT chứa Embeddings CỦA TẬP TRAIN
        data_root="./data/data",
        num_sampled_frames=8,
        vis_dir="./visualizations_train",
        num_vis=20,       
        max_samples=20    
    )
    
    benchmark.evaluate()