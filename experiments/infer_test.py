import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
from einops import rearrange

# Import model từ file train
from experiments.train_test import CustomVideoBackbone, LiMaVLM

# ==========================================
# 1. HÀM DECODE CENTERPOINT (Giữ nguyên)
# ==========================================
def decode_centerpoint(hm, sz, off, k=5):
    """
    Biến Heatmap thành Bounding Boxes.
    hm: [1, 1, 48, 48], sz: [1, 2, 48, 48], off: [1, 2, 48, 48]
    """
    batch, cat, height, width = hm.size()
    
    keep = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (keep == hm).float()
    hm = hm * keep
    
    scores, inds = torch.topk(hm.view(batch, -1), k)
    
    topk_ys = (inds // width).float()
    topk_xs = (inds % width).float()
    
    off = rearrange(off, 'b c h w -> b (h w) c')
    sz = rearrange(sz, 'b c h w -> b (h w) c')
    
    batch_inds = torch.arange(batch).view(-1, 1).to(inds.device)
    topk_off = off[batch_inds, inds] 
    topk_sz = sz[batch_inds, inds]   
    
    xs = topk_xs + topk_off[..., 0]
    ys = topk_ys + topk_off[..., 1]
    
    w = torch.exp(topk_sz[..., 0])
    h = torch.exp(topk_sz[..., 1])
    
    bboxes = torch.stack([
        scores,
        (xs - w/2) / width, (ys - h/2) / height,
        (xs + w/2) / width, (ys + h/2) / height
    ], dim=-1)
    
    return bboxes[0] 

# ==========================================
# 2. HÀM INFERENCE XUẤT VIDEO
# ==========================================
def run_inference(video_id, checkpoint_path, data_root="./data/data", json_path="./data/data/train-tracks.json", output_video="inference_result.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 384
    
    # --- Load Model ---
    backbone = CustomVideoBackbone(d_model=256, num_frames=8, img_size=img_size, patch_size=8).to(device)
    
    # 🚀 SỬA Ở ĐÂY 1: Khai báo đúng num_blocks=2 giống lúc cấu hình file train
    vlm_head = LiMaVLM(d_model=256, d_text=512, num_blocks=2).to(device)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(ckpt['backbone_state'])
    # Dùng strict=False đề phòng các phiên bản trước có lưu thêm/bớt key
    vlm_head.load_state_dict(ckpt['vlm_state'], strict=False)
    backbone.eval(); vlm_head.eval()

    # --- Load Data Info ---
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    track_info = raw_data[video_id]
    frames_list = track_info["frames"]
    boxes_list = track_info["boxes"] 
    text_query = track_info["nl"][0]
    
    print("="*50)
    print(f"Query: {text_query}")
    print(f"Processing {len(frames_list)} frames...")
    print("="*50)

    # Load file Tokens
    text_embs = torch.load(os.path.join(data_root, "clip_text_tokens.pt"), map_location=device)
    text_tokens = text_embs[text_query.strip().lower()].to(device).unsqueeze(0).float() # Shape: [1, 32, 512]

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    processed_frames = [None] * len(frames_list)

    # --- Xử lý Video ---
    for start in range(0, len(frames_list) - 8 + 1, 4):
        chunk_frames = frames_list[start : start + 8]
        tensors, orig_imgs = [], []
        
        for p in chunk_frames:
            # Chuyển đổi nhanh bằng CV2 thay vì PIL nếu có thể, hoặc giữ nguyên PIL như cũ
            img_raw = Image.open(os.path.join(data_root, p)).convert('RGB')
            orig_imgs.append(np.array(img_raw))
            tensors.append(transform(img_raw))
        
        input_v = torch.stack(tensors, dim=1).unsqueeze(0).to(device)

        with torch.no_grad():
            # 🚀 SỬA Ở ĐÂY 2: Bật không gian bfloat16 để tránh nổ số khi qua Mamba
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                feats = backbone(input_v)
                p_hm, p_sz, p_off = vlm_head(feats, text_tokens)
            
            # Ép lại về Float32 để tính toán vẽ Box không bị sai số
            p_hm = p_hm.float()
            p_sz = p_sz.float()
            p_off = p_off.float()
            
        for i in range(8):
            global_idx = start + i
            img_draw = cv2.cvtColor(orig_imgs[i], cv2.COLOR_RGB2BGR)
            h_orig, w_orig = img_draw.shape[:2]
            
            # ==========================================
            # 🚀 CHÈN CODE VẼ HEATMAP Ở ĐÂY
            # ==========================================
            # 1. Trích xuất Heatmap của frame hiện tại (Shape: 48x48)
            hm_frame = p_hm[i, 0].cpu().numpy() 
            
            # 2. Chuẩn hóa giá trị từ [0, 1] sang thang độ xám [0, 255]
            hm_norm = np.clip(hm_frame * 255, 0, 255).astype(np.uint8)
            
            # 3. Phóng to Heatmap lên bằng kích thước ảnh gốc (Dùng Cubic để mượt)
            hm_resized = cv2.resize(hm_norm, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
            
            # 4. Phủ màu nhiệt (JET: Đỏ = tự tin cao, Xanh dương = tự tin thấp)
            hm_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
            
            # 5. Chồng Heatmap lên ảnh gốc (Tỉ lệ: 60% ảnh gốc, 40% Heatmap)
            img_draw = cv2.addWeighted(img_draw, 0.6, hm_color, 0.4, 0)
            # ==========================================

            # --- VẼ GROUND TRUTH (MÀU ĐỎ) ---
            gt_box = boxes_list[global_idx] 
            gt_x, gt_y, gt_w, gt_h = gt_box
            gx1, gy1 = int(gt_x), int(gt_y)
            gx2, gy2 = int(gt_x + gt_w), int(gt_y + gt_h)
            
            cv2.rectangle(img_draw, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
            cv2.putText(img_draw, "GT", (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # --- VẼ PREDICTION (MÀU XANH LÁ) ---
            dets = decode_centerpoint(p_hm[i:i+1], p_sz[i:i+1], p_off[i:i+1], k=5)

            for d in range(5):
                score, x1, y1, x2, y2 = dets[d].cpu().numpy()
                if score > 0.01: 
                    # TÍNH TOÁN LẠI TỌA ĐỘ CHO TỪNG ỨNG VIÊN
                    ix1, iy1 = int(x1 * w_orig), int(y1 * h_orig)
                    ix2, iy2 = int(x2 * w_orig), int(y2 * h_orig)
                    
                    # Vẽ màu nhạt dần: Score cao = Xanh sáng, Score thấp = Xanh tối
                    color = (0, int(255 * score), 0) 
                    cv2.rectangle(img_draw, (ix1, iy1), (ix2, iy2), color, 1)
                    
                    # Chỉ hiện text cho những thằng có score tương đối để tránh rối mắt
                    if score > 0.05:
                        cv2.putText(img_draw, f"{score:.2f}", (ix1, iy1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            processed_frames[global_idx] = img_draw


    # --- LƯU RA FILE MP4 ---
    print(f"Saving video to {output_video}...")
    
    first_valid_frame = next(f for f in processed_frames if f is not None)
    h_target, w_target = first_valid_frame.shape[:2]
    
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w_target, h_target))

    written_count = 0
    for frame in processed_frames:
        if frame is not None:
            if frame.shape[:2] != (h_target, w_target):
                frame = cv2.resize(frame, (w_target, h_target))
            out.write(frame)
            written_count += 1

    out.release()
    print("="*50)
    print(f"✅ Inference Finished Successfully!")
    print(f"📊 Total frames processed & saved: {written_count} / {len(frames_list)}")
    print(f"⏱️ Video duration: {written_count / fps:.2f} seconds.")
    print("="*50)

# ==========================================
# 3. RUN
# ==========================================
if __name__ == "__main__":
    run_inference(
        video_id="efa486f0-986b-4adb-8ff2-07ff6314bf3c", 
        checkpoint_path="./checkpoints/limavlm_best.pth",
        output_video="inference_result.mp4"
    )