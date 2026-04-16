import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

# Import kiến trúc mới của bạn
from models.lima import LiMaVLM

# ==========================================
# 1. HÀM DECODE CENTERNET (Chuẩn hóa [0, 1])
# ==========================================
def decode_centernet_topk(hm, sz, off, k=5):
    """
    Biến Heatmap thành Bounding Boxes theo kiến trúc mới (không dùng exp).
    hm: [B, 1, H, W], sz: [B, 2, H, W], off: [B, 2, H, W]
    Trả về tọa độ đã chuẩn hóa [0, 1] để dễ nhân với ảnh gốc.
    """
    B, C, H, W = hm.shape
    
    # 1. Non-Maximum Suppression (NMS)
    keep = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (keep == hm).float()
    hm = hm * keep
    
    # 2. Lấy Top-K ứng viên
    scores, inds = torch.topk(hm.view(B, -1), k)
    
    ys = (inds // W).int().float()
    xs = (inds % W).int().float()
    
    # Flatten spatial dims
    off = off.view(B, 2, -1)
    sz = sz.view(B, 2, -1)
    
    bboxes_list = []
    for i in range(B):
        frame_bboxes = []
        for j in range(k):
            idx = inds[i, j].item()
            score = scores[i, j].item()
            y, x = ys[i, j].item(), xs[i, j].item()
            
            offset_x = off[i, 0, idx].item()
            offset_y = off[i, 1, idx].item()
            
            # Theo kiến trúc trước: sz[0] là chiều cao (h), sz[1] là chiều rộng (w)
            height_box = sz[i, 0, idx].item()
            width_box = sz[i, 1, idx].item()
            
            cx = x + offset_x
            cy = y + offset_y
            
            # Chuẩn hóa về [0, 1] dựa trên kích thước Feature Map (H, W)
            x1 = (cx - width_box/2) / W
            y1 = (cy - height_box/2) / H
            x2 = (cx + width_box/2) / W
            y2 = (cy + height_box/2) / H
            
            frame_bboxes.append([score, x1, y1, x2, y2])
        bboxes_list.append(frame_bboxes)
        
    return torch.tensor(bboxes_list) # Shape: [B, k, 5]

# ==========================================
# 2. HÀM INFERENCE XUẤT VIDEO
# ==========================================
def run_inference(video_id, checkpoint_path, data_root="./data/data", json_path="./data/data/train-tracks.json", output_video="inference_result.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 384
    chunk_size = 8 # Số frame xử lý mỗi lần để đẩy vào Mamba
    
    # --- Load Model (Kiến trúc Mới) ---
    print(f"🧠 Đang khởi tạo LiMaVLM...")
    model = LiMaVLM(d_model=256, d_text=512, num_blocks=3).to(device)
    
    print(f"📦 Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Lấy state_dict nếu lưu toàn bộ model, bỏ qua strict=False để an toàn
    model.load_state_dict(ckpt if 'model_state' not in ckpt else ckpt['model_state'], strict=False)
    model.eval()

    # --- Load Data Info ---
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    track_info = raw_data[video_id]
    frames_list = track_info["frames"]
    boxes_list = track_info["boxes"] 
    text_query = track_info["nl"][0]
    
    print("="*50)
    print(f"📝 Query: {text_query}")
    print(f"🎞️ Processing {len(frames_list)} frames...")
    print("="*50)

    # Load file Tokens Text
    text_embs = torch.load(os.path.join(data_root, "clip_text_tokens.pt"), map_location=device)
    # Trích xuất đúng vector của câu Query
    text_tokens = text_embs[text_query.strip().lower()].to(device).unsqueeze(0).float() # [1, 32, 512]

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    processed_frames = []
    states = None # 🚀 KHỞI TẠO BỘ NHỚ CHO MAMBA

    # --- Xử lý Video (Chạy chunk nối tiếp nhau để truyền states) ---
    for start in range(0, len(frames_list), chunk_size):
        end = min(start + chunk_size, len(frames_list))
        chunk_frames = frames_list[start:end]
        
        tensors, orig_imgs = [], []
        
        for p in chunk_frames:
            img_raw = Image.open(os.path.join(data_root, p)).convert('RGB')
            orig_imgs.append(np.array(img_raw))
            tensors.append(transform(img_raw))
        
        # input_v shape: [1, 3, T, H, W]
        input_v = torch.stack(tensors, dim=1).unsqueeze(0).to(device)

        with torch.no_grad():
            # Bật không gian bfloat16 để tránh nổ số khi qua Mamba
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_v, text_tokens, states=states)
            
            # Cập nhật states cho chunk tiếp theo
            states = outputs["memory_states"]
            
            # Ép lại về Float32
            p_hm, p_sz, p_off = [x.float() for x in outputs["tracking_heads"]]
            
        current_chunk_len = len(chunk_frames)
        
        # LiMaVLM trả về heads dưới dạng gộp Batch*Time, ta tách ra lại
        # Hình dạng: [chunk_len, C, H, W]
        p_hm = p_hm.view(current_chunk_len, 1, p_hm.size(-2), p_hm.size(-1))
        p_sz = p_sz.view(current_chunk_len, 2, p_sz.size(-2), p_sz.size(-1))
        p_off = p_off.view(current_chunk_len, 2, p_off.size(-2), p_off.size(-1))
            
        # Decode lấy Top-5 boxes
        dets_batch = decode_centernet_topk(p_hm, p_sz, p_off, k=5)
            
        for i in range(current_chunk_len):
            global_idx = start + i
            img_draw = cv2.cvtColor(orig_imgs[i], cv2.COLOR_RGB2BGR)
            h_orig, w_orig = img_draw.shape[:2]
            
            # ==========================================
            # 🚀 VẼ HEATMAP TỪ BẢN CŨ
            # ==========================================
# ==========================================
            # 🚀 VẼ HEATMAP (GIỮ NGUYÊN MÀU GỐC CỦA VIDEO)
            # ==========================================
            hm_frame = p_hm[i, 0].cpu().numpy() 
            
            # 1. Phóng to heatmap [0, 1] lên bằng kích thước ảnh gốc
            hm_resized = cv2.resize(hm_frame, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
            hm_resized = np.clip(hm_resized, 0, 1) # Đảm bảo giá trị an toàn trong khoảng 0 -> 1
            
            # 2. Nhân bản thành 3 kênh để khớp với ảnh RGB
            hm_mask = np.stack([hm_resized, hm_resized, hm_resized], axis=-1)
            
            # 3. Trộn Mask với ảnh gốc:
            # - base_brightness = 0.3: Vùng background (không có xe) sẽ bị tối đi, chỉ sáng bằng 30% ảnh gốc.
            # - hm_mask: Cộng thêm độ sáng cho vùng có xe (tối đa lên 100% màu gốc).
            base_brightness = 0.3
            final_mask = np.clip(hm_mask + base_brightness, 0, 1)
            
            # 4. Áp dụng lên ảnh (Nhân giá trị pixel gốc với mặt nạ)
            img_draw = (img_draw.astype(np.float32) * final_mask).astype(np.uint8)
            # ==========================================
            
            # ==========================================
            # 🚀 VẼ GROUND TRUTH (MÀU ĐỎ)
            # ==========================================
            gt_box = boxes_list[global_idx] 
            gt_x, gt_y, gt_w, gt_h = gt_box
            gx1, gy1 = int(gt_x), int(gt_y)
            gx2, gy2 = int(gt_x + gt_w), int(gt_y + gt_h)
            
            cv2.rectangle(img_draw, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
            cv2.putText(img_draw, "GT", (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # ==========================================
            # 🚀 VẼ PREDICTION (MÀU XANH LÁ - FADING)
            # ==========================================
            dets = dets_batch[i] # [5, 5] (score, x1, y1, x2, y2)
            
            for d in range(5):
                score, x1, y1, x2, y2 = dets[d].numpy()
                if score > 0.01: 
                    # Trả tọa độ chuẩn hóa về kích thước ảnh thật
                    ix1, iy1 = int(x1 * w_orig), int(y1 * h_orig)
                    ix2, iy2 = int(x2 * w_orig), int(y2 * h_orig)
                    
                    # Vẽ màu nhạt dần: Score cao = Xanh sáng, Score thấp = Xanh tối
                    color = (0, int(255 * score), 0) 
                    cv2.rectangle(img_draw, (ix1, iy1), (ix2, iy2), color, 2)
                    
                    if score > 0.05:
                        cv2.putText(img_draw, f"{score:.2f}", (ix1, iy1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # In Text Query lên video
            cv2.putText(img_draw, f"Query: {text_query}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            processed_frames.append(img_draw)

    # --- LƯU RA FILE MP4 ---
    print(f"🎬 Saving video to {output_video}...")
    
    h_target, w_target = processed_frames[0].shape[:2]
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w_target, h_target))

    for frame in processed_frames:
        out.write(frame)

    out.release()
    print("="*50)
    print(f"✅ Inference Finished Successfully!")
    print(f"📊 Total frames processed & saved: {len(processed_frames)}")
    print(f"⏱️ Video duration: {len(processed_frames) / fps:.2f} seconds.")
    print("="*50)

# ==========================================
# 3. RUN
# ==========================================
if __name__ == "__main__":
    run_inference(
        video_id="f6b8685c-9eb1-47f4-bd22-3c517ec56767", # Đổi UUID test của bạn ở đây
        checkpoint_path="checkpoint_epoch_80.pth", 
        output_video="inference_heatmaps_result.mp4"
    )