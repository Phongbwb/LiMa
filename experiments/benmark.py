import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.ops as ops
from torchvision import transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import models
from models.lima import LiMaVLM
from models.microlocal import MicroLocalization
# Đảm bảo đường dẫn import LNN phù hợp với source code của bạn
from models.astropoolv2 import LNNBboxComparator 

# ==========================================================
# 1. CÁC HÀM PHỤ TRỢ
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
# 2. PIPELINE SUY LUẬN (LIMA -> LNN + MICRO)
# ==========================================================
class VideoPartGroundingPipeline:
    def __init__(self, lima_ckpt_path, micro_ckpt_path, lnn_ckpt_path, device='cuda'):
        self.device = device
        self.lima_downsample_ratio = 16 
        
        # 1. Tải LiMaVLM
        print("Đang tải LiMaVLM...")
        self.lima_model = LiMaVLM(d_model=256, d_text=512, num_blocks=4).to(self.device)
        lima_state = torch.load(lima_ckpt_path, map_location=self.device)
        self.lima_model.load_state_dict(lima_state.get('model_state_dict', lima_state))
        self.lima_model.eval()

        # 2. Tải MicroLocalization
        print("Đang tải MicroLocalization...")
        self.micro_model = MicroLocalization(embed_dim=512, num_parts=5, roi_size=7).to(self.device)
        micro_state = torch.load(micro_ckpt_path, map_location=self.device)
        self.micro_model.load_state_dict(micro_state.get('model_state_dict', micro_state))
        self.micro_model.eval()

        # 3. Tải LNN Model
        print("Đang tải LNNBboxComparator...")
        self.lnn_model = LNNBboxComparator(text_dim=512, hidden_dim=256).to(self.device)
        lnn_state = torch.load(lnn_ckpt_path, map_location=self.device)
        self.lnn_model.load_state_dict(lnn_state.get('model_state_dict', lnn_state))
        self.lnn_model.eval()

    def _crop_and_resize_gpu(self, video_frames_512, bboxes_cxcywh):
        B, T, _ = bboxes_cxcywh.shape
        
        if video_frames_512.dim() == 5:
            if video_frames_512.shape[1] == T: 
                frames_flat = video_frames_512.reshape(B * T, 3, 384, 384)
            else: 
                frames_flat = video_frames_512.permute(0, 2, 1, 3, 4).reshape(B * T, 3, 384, 384)
        else:
            frames_flat = video_frames_512

        boxes_384 = bboxes_cxcywh * self.lima_downsample_ratio
        cx, cy, w, h = boxes_384[..., 0], boxes_384[..., 1], boxes_384[..., 2], boxes_384[..., 3]
        
        x1 = torch.clamp(cx - w / 2, min=0)
        y1 = torch.clamp(cy - h / 2, min=0)
        x2 = torch.clamp(cx + w / 2, max=384)
        y2 = torch.clamp(cy + h / 2, max=384)
        
        x2 = torch.where(x2 <= x1, x1 + 10, x2)
        y2 = torch.where(y2 <= y1, y1 + 10, y2)
        
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).view(B * T, 4) 
        
        batch_inds = torch.arange(B * T, device=self.device).unsqueeze(1).float()
        rois = torch.cat([batch_inds, boxes_xyxy], dim=1)
        
        cropped_rois = ops.roi_align(frames_flat, rois, output_size=(336, 336), spatial_scale=1.0)
        return cropped_rois

    @torch.no_grad()
    def run_inference(self, video_frames, color_text, type_text, motion_text, context_text):
        # 1. Chạy LiMaVLM để lấy 5 BBox Candidates
        lima_output = self.lima_model(
            video_frames=video_frames,
            color_text_tokens=color_text,
            type_text_tokens=type_text,
            motion_text_tokens=motion_text,
            context_text_tokens=context_text
        )
        
        lima_bboxes = lima_output["final_bboxes"] # [1, T, 5, 4]
        frame_scores = lima_output["frame_scores"] # [1, T]

        _, T, num_cands, _ = lima_bboxes.shape
        
        final_best_bboxes = []
        video_level_scores = []
        
        trajectory_window = [] 

        # Đánh giá qua từng Frame
        for t in range(T):
            cand_bboxes = lima_bboxes[0, t] # [5, 4] -> Chứa [cx, cy, w, h]
            
            # --- A. LNN EVALUATION (Song song 5 candidates) ---
            lnn_inputs = []
            for k in range(num_cands):
                cand = cand_bboxes[k]
                
                # Tính vận tốc dx, dy
                if len(trajectory_window) == 0:
                    dx, dy = 0.0, 0.0
                else:
                    prev_best_box = trajectory_window[-1] 
                    dx = cand[0].item() - prev_best_box[0].item()
                    dy = cand[1].item() - prev_best_box[1].item()
                
                velocity_tensor = torch.tensor([dx, dy], dtype=cand.dtype, device=self.device)
                cand_6d = torch.cat([cand, velocity_tensor])
                
                if len(trajectory_window) == 0:
                    seq = [cand_6d, cand_6d, cand_6d]
                elif len(trajectory_window) == 1:
                    seq = [trajectory_window[0], cand_6d, cand_6d]
                else:
                    seq = [trajectory_window[0], trajectory_window[1], cand_6d]
                
                lnn_inputs.append(torch.stack(seq))
                
            lnn_inputs_tensor = torch.stack(lnn_inputs) # [5, 3, 6]
            lnn_text = motion_text.expand(num_cands, -1, -1) # [5, N, 512]
            
            # Tính điểm LNN
            lnn_scores = self.lnn_model(lnn_inputs_tensor, lnn_text, dt=1.0)
            lnn_scores = lnn_scores.view(num_cands, -1).mean(dim=1) 
            
            # --- B. MICRO LOCALIZATION EVALUATION ---
            frame_t = video_frames[:, :, t:t+1, :, :] 
            frame_t_expanded = frame_t.repeat(num_cands, 1, 1, 1, 1) # [5, 1, C, H, W]
            cand_bboxes_unsqueeze = cand_bboxes.unsqueeze(1) 
            
            rois = self._crop_and_resize_gpu(frame_t_expanded, cand_bboxes_unsqueeze) 
            
            color_text_exp = color_text.expand(num_cands, -1, -1)
            type_text_exp = type_text.expand(num_cands, -1, -1)
            
            _, _, _, _, roi_color, global_color = self.micro_model(rois, word_tokens=color_text_exp)
            _, _, _, _, roi_type, global_type = self.micro_model(rois, word_tokens=type_text_exp)
            
            micro_score_color = roi_color.view(num_cands, -1).mean(dim=1) + global_color.view(num_cands, -1).mean(dim=1)
            micro_score_type = roi_type.view(num_cands, -1).mean(dim=1) + global_type.view(num_cands, -1).mean(dim=1)
            
            micro_scores = (micro_score_color + micro_score_type) / 2.0
            
            # --- C. TỔNG HỢP VÀ CHỌN BBOX TỐT NHẤT ---
            lima_scores = frame_scores[0, t] 
            
            total_scores = 0.05 * lnn_scores + 0.5 * micro_scores + 1.0 * lima_scores
            
            best_idx = total_scores.argmax().item() 
            best_cand = cand_bboxes[best_idx]
            
            best_score = total_scores[best_idx].item()
            
            final_best_bboxes.append(best_cand)
            video_level_scores.append(best_score)
            
            # Cập nhật cửa sổ trượt
            best_cand_6d = torch.cat([best_cand, torch.tensor([1.0, 1.0], device=self.device)])
            trajectory_window.append(best_cand_6d)
            if len(trajectory_window) > 2:
                trajectory_window.pop(0)

        final_best_bboxes = torch.stack(final_best_bboxes).unsqueeze(0) # [1, T, 4]
        best_cand_scores = torch.tensor(video_level_scores, device=self.device) # Lấy mảng điểm tự tin từng frame
        final_video_score = best_cand_scores.max()
        
        # Heatmap Trích xuất
        attention_map_2d, backbone_map_2d = None, None
        last_attn_layer = None
        for module in self.lima_model.modules():
            if module.__class__.__name__ == 'VisionTextCrossAttention':
                last_attn_layer = module
        if last_attn_layer is not None and hasattr(last_attn_layer, 'last_attn_weights'):
            attn_w = last_attn_layer.last_attn_weights
            attn_spatial = attn_w.mean(dim=-1) 
            seq_len = attn_spatial.shape[-1]
            H_feat = W_feat = int(np.sqrt(seq_len))
            attention_map_2d = attn_spatial.view(1, T, H_feat, W_feat).cpu()

        if "debug_feat_map" in lima_output and lima_output["debug_feat_map"] is not None:
            bb_feat = lima_output["debug_feat_map"].detach().cpu()
            bb_feat_mean = bb_feat.mean(dim=1) 
            _, H_bb, W_bb = bb_feat_mean.shape
            backbone_map_2d = bb_feat_mean.view(1, T, H_bb, W_bb)

        return {
            "final_score": final_video_score,
            "lima_bboxes": final_best_bboxes,
            "best_cand_scores": best_cand_scores, # TRẢ VỀ ĐIỂM CỦA BEST CANDIDATE ĐỂ VISUALIZE
            "frame_scores": frame_scores,
            "attn_heatmap": attention_map_2d,
            "backbone_heatmap": backbone_map_2d
        }

# ==========================================================
# 3. LỚP QUẢN LÝ BENCHMARK
# ==========================================================
class CityFlowDirectBenchmark:
    def __init__(self, pipeline, queries_path, tracks_path, text_emb_path, data_root, num_sampled_frames=8, vis_dir="./visualizations", num_vis=5, max_samples=20): 
        self.pipeline = pipeline
        self.num_sampled_frames = num_sampled_frames
        self.device = pipeline.device
        self.data_root = data_root
        self.vis_dir = vis_dir
        self.num_vis = num_vis
        self.max_samples = max_samples 
        os.makedirs(self.vis_dir, exist_ok=True)
        
        print("📥 Đang tải dữ liệu JSON và Đặc trưng văn bản...")
        with open(queries_path, 'r') as f: self.queries_data = json.load(f)
        with open(tracks_path, 'r') as f: self.tracks_data = json.load(f)
            
        self.text_features_dict = torch.load(text_emb_path, map_location=self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.cached_videos = {}

    def preload_videos(self, track_keys):
        print(f"🔄 Đang tiền xử lý {len(track_keys)} Video Tracks...")
        for track_uuid in tqdm(track_keys, desc="Loading Videos"):
            track_info = self.tracks_data[track_uuid]
            total_frames = len(track_info['frames'])
            indices = np.linspace(0, total_frames - 1, self.num_sampled_frames, dtype=int)
            
            sampled_frames, sampled_boxes, orig_frames_list = [], [], []
            orig_size = None      
            
            for idx in indices:
                img_path = os.path.join(self.data_root, track_info['frames'][idx].replace('./', ''))
                img = Image.open(img_path).convert("RGB")
                
                if orig_size is None: orig_size = img.size 
                scale_w, scale_h = 384.0 / orig_size[0], 384.0 / orig_size[1]
                
                orig_frames_list.append(img.copy())
                sampled_frames.append(self.transform(img))
                
                ob = track_info['boxes'][idx]
                sampled_boxes.append([ob[0]*scale_w, ob[1]*scale_h, ob[2]*scale_w, ob[3]*scale_h])
                
            video_tensor = torch.stack(sampled_frames, dim=1).unsqueeze(0).to(self.device)
            gt_boxes = torch.tensor(sampled_boxes, dtype=torch.float32).to(self.device)
            self.cached_videos[track_uuid] = (video_tensor, gt_boxes, orig_frames_list, orig_size)

    # Thêm score_threshold vào đây
    def visualize_predictions(self, query_text, orig_frames_list, orig_size, gt_boxes, pred_boxes, save_path, iou_score, pred_scores=None, attn_heatmap=None, backbone_heatmap=None, score_threshold=0.3):
        T = len(orig_frames_list)
        orig_w, orig_h = orig_size
        scale_x, scale_y = orig_w / 384.0, orig_h / 384.0

        fig, axes = plt.subplots(2, T, figsize=(4 * T, 8)) 
        fig.suptitle(f"Query: {query_text}\nmIoU (Top-1 Selected): {iou_score:.4f}", fontsize=16, fontweight='bold')

        box_color = 'lime'

        for t in range(T):
            # Row 1: Ảnh bình thường (Normal Image) + Bbox (Có điểm, qua threshold)
            ax1 = axes[0, t] if T > 1 else axes[0]
            ax1.imshow(orig_frames_list[t])
            ax1.axis('off')

            if gt_boxes is not None and len(gt_boxes) > t:
                gx, gy, gw, gh = gt_boxes[t].numpy()
                gx, gy, gw, gh = gx * scale_x, gy * scale_y, gw * scale_x, gh * scale_y
                ax1.add_patch(patches.Rectangle((gx, gy), gw, gh, linewidth=2, edgecolor='white', facecolor='none', linestyle=':'))

            # Xử lý Threshold & Confidence Score cho predicted bbox
            px, py, pw, ph = pred_boxes[t].numpy()
            score = pred_scores[t].item() if pred_scores is not None else 1.0

            if score >= score_threshold: # Chỉ vẽ nếu độ tự tin cao hơn ngưỡng
                px, py, pw, ph = px * scale_x, py * scale_y, pw * scale_x, ph * scale_y
                px1, py1 = px - pw / 2, py - ph / 2 
                
                ax1.add_patch(patches.Rectangle((px1, py1), pw, ph, linewidth=3, edgecolor=box_color, facecolor='none', linestyle='-'))
                # In text điểm độ tự tin
                ax1.text(px1, py1 - 5, f"{score:.2f}", color='black', fontsize=12, fontweight='bold', bbox=dict(facecolor=box_color, alpha=0.8, edgecolor='none', pad=1))

            # Row 2: Heatmap 
            ax2 = axes[1, t] if T > 1 else axes[1]
            ax2.imshow(orig_frames_list[t])
            ax2.axis('off')

            if backbone_heatmap is not None:
                bb_hm = backbone_heatmap[t]
                bb_norm = (bb_hm - bb_hm.min()) / (bb_hm.max() - bb_hm.min() + 1e-8)
                bb_resized = F.interpolate(bb_norm.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode='bicubic', align_corners=False).squeeze().numpy()
                ax2.imshow(plt.get_cmap('viridis')(bb_resized)[..., :3], alpha=0.6) 
            elif attn_heatmap is not None:
                # Fallback vẽ attn_heatmap ở row 2 nếu backbone_heatmap ko có
                hm = attn_heatmap[t]
                hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
                hm_resized = F.interpolate(hm_norm.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode='bicubic', align_corners=False).squeeze().numpy()
                ax2.imshow(plt.get_cmap('jet')(hm_resized)[..., :3], alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
    def evaluate(self):
        query_keys = list(self.queries_data.keys())[:self.max_samples]
        track_keys = list(self.tracks_data.keys())[:self.max_samples]
        num_pairs, num_tracks = len(query_keys), len(track_keys)
        
        self.preload_videos(track_keys)
        
        score_matrix = np.zeros((num_pairs, num_tracks))
        ranks = np.zeros(num_pairs)
        all_ious = []

        print(f"🚀 Bắt đầu Benchmark: Đánh giá {num_pairs} cặp...")
        
        # Ngưỡng (Threshold) cấu hình ở đây - bạn có thể tinh chỉnh 
        confidence_threshold = 0.3 

        with torch.no_grad():
            for i in tqdm(range(num_pairs), desc="Đánh giá Queries"):
                q_key = query_keys[i]
                query_text = self.queries_data[q_key]['nl'][0].strip().lower()
                text_data = self.text_features_dict.get(query_text, None)
                
                if text_data is None:
                    ranks[i] = num_tracks 
                    continue
                    
                motion_txt = text_data["motion_embedding"].unsqueeze(0).to(self.device, non_blocking=True)
                color_txt = text_data["color_embedding"].unsqueeze(0).to(self.device, non_blocking=True)
                type_txt = text_data["type_embedding"].unsqueeze(0).to(self.device, non_blocking=True)
                context_txt = text_data["context_embedding"].unsqueeze(0).to(self.device, non_blocking=True)
                
                for j, t_key in enumerate(track_keys):
                    video_tensor, gt_boxes, orig_frames_list, orig_size = self.cached_videos[t_key]
                    
                    results = self.pipeline.run_inference(
                        video_frames=video_tensor, 
                        color_text=color_txt, type_text=type_txt,
                        motion_text=motion_txt, context_text=context_txt
                    )
                    
                    score_matrix[i, j] = results["final_score"].item()
                    
                    if j == i: 
                        pred_bboxes = results["lima_bboxes"].squeeze(0) * self.pipeline.lima_downsample_ratio
                        pred_scores = results["best_cand_scores"] # Lấy scores
                        
                        iou = calculate_iou_batch(pred_bboxes, gt_boxes)
                        mean_iou_val = iou.mean().item()
                        all_ious.append(mean_iou_val)

                        if i < self.num_vis:
                            save_path = os.path.join(self.vis_dir, f"query_vis_{i:03d}.png")
                            attn_heatmap_cpu = results["attn_heatmap"].squeeze(0) if results["attn_heatmap"] is not None else None
                            bb_heatmap_cpu = results["backbone_heatmap"].squeeze(0) if results["backbone_heatmap"] is not None else None

                            self.visualize_predictions(
                                query_text=query_text, 
                                orig_frames_list=orig_frames_list, orig_size=orig_size, 
                                gt_boxes=gt_boxes.cpu(), 
                                pred_boxes=pred_bboxes.cpu(), 
                                save_path=save_path, iou_score=mean_iou_val,
                                pred_scores=pred_scores.cpu(), # Truyền mảng điểm
                                attn_heatmap=attn_heatmap_cpu,
                                backbone_heatmap=bb_heatmap_cpu,
                                score_threshold=confidence_threshold # Truyền Threshold
                            )

                sorted_indices = np.argsort(-score_matrix[i, :])
                ranks[i] = np.where(sorted_indices == i)[0][0] + 1 

        print("\n" + "="*50)
        print("🏆 KẾT QUẢ CITYFLOW-NL (LIMA + LNN + MICRO) 🏆")
        print("="*50)
        print(f"🔹 Mean Reciprocal Rank (MRR) : {np.sum(1.0 / ranks) / num_pairs:.4f}")
        print(f"🔹 Retrieval Recall@1         : {np.sum(ranks == 1) / num_pairs * 100:.2f}%")
        print(f"🔹 Retrieval Recall@5         : {np.sum(ranks <= 5) / num_pairs * 100:.2f}%")
        print(f"🔹 Retrieval Recall@10        : {np.sum(ranks <= 10) / num_pairs * 100:.2f}%")
        print(f"🔹 Localization mIoU          : {np.mean(all_ious) if len(all_ious) > 0 else 0.0:.4f}")
        print("="*50)

# ==========================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==========================================================
if __name__ == "__main__":
    my_pipeline = VideoPartGroundingPipeline(
        lima_ckpt_path="./data/data/checkpoint_stage1_epoch_8_1.pth",
        micro_ckpt_path="./checkpoints/micro_localization/best_model_epoch_26.pth",
        lnn_ckpt_path="./checkpoints/lnn_bbox_comparator/lnn_bbox_epoch_79.pth", 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    benchmark = CityFlowDirectBenchmark(
        pipeline=my_pipeline, 
        queries_path="./data/data/test-queries.json", 
        tracks_path="./data/data/test-tracks.json",
        text_emb_path="./data/data/clip_text_tokens_extracted_optimized.pt",
        data_root="./data/data",
        num_sampled_frames=8,
        vis_dir="./visualizations",
        num_vis=100,      
        max_samples=120  
    )
    
    benchmark.evaluate()