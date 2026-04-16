import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPTokenizer, CLIPTextModel

# Import kiến trúc mô hình của bạn
from models.lima import LiMaVLM

# ==========================================
# 1. HÀM GIẢI MÃ CENTERNET
# ==========================================
def decode_centernet_topk(hm, sz, off, k=5):
    """Giải mã Heatmap thành Bounding Box (Tọa độ chuẩn hóa [0, 1])"""
    B, C, H, W = hm.shape
    
    keep = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (keep == hm).float()
    hm = hm * keep
    
    scores, inds = torch.topk(hm.view(B, -1), k)
    ys, xs = (inds // W).int().float(), (inds % W).int().float()
    off, sz = off.view(B, 2, -1), sz.view(B, 2, -1)
    
    bboxes_list = []
    for i in range(B):
        frame_bboxes = []
        for j in range(k):
            idx, score = inds[i, j].item(), scores[i, j].item()
            y, x = ys[i, j].item(), xs[i, j].item()
            
            cx = x + off[i, 0, idx].item()
            cy = y + off[i, 1, idx].item()
            height_box, width_box = sz[i, 0, idx].item(), sz[i, 1, idx].item()
            
            x1, y1 = (cx - width_box/2) / W, (cy - height_box/2) / H
            x2, y2 = (cx + width_box/2) / W, (cy + height_box/2) / H
            
            frame_bboxes.append([score, x1, y1, x2, y2])
        bboxes_list.append(frame_bboxes)
        
    return torch.tensor(bboxes_list)

# ==========================================
# 2. HÀM XỬ LÝ TEXT TỰ NHẬP BẰNG CLIP
# ==========================================
def encode_custom_text(text, device='cuda'):
    """Biến text tùy ý thành tensor [1, 32, 512] bằng OpenAI CLIP"""
    print("🔤 Đang tải Text Encoder (CLIP)...")
    model_id = "openai/clip-vit-base-patch32" # Model gốc xuất ra d_model=512
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_model = CLIPTextModel.from_pretrained(model_id).to(device)
    text_model.eval()
    
    # Ép max_length=32 để khớp với kiến trúc LiMaVLM của bạn
    inputs = tokenizer(
        text, padding="max_length", max_length=32, 
        truncation=True, return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = text_model(**inputs.to(device))
        text_tokens = outputs.last_hidden_state # Shape: [1, 32, 512]
        
    return text_tokens.float()

# ==========================================
# 3. HÀM INFERENCE CHÍNH CỦA BẠN
# ==========================================
def run_custom_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 384
    chunk_size = args.chunk_size
    
    # 1. Khởi tạo và encode Text
    print(f"📝 Text tự nhập: '{args.query_text}'")
    text_tokens = encode_custom_text(args.query_text, device)
    
    # 2. Đọc Video tùy ý
    print(f"🎞️ Đang đọc video từ: {args.video_path}")
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Không thể mở video: {args.video_path}")
        
    orig_frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        # OpenCV mặc định là BGR, chuyển sang RGB
        orig_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    total_frames = len(orig_frames)
    print(f"✅ Đã tải xong video ({total_frames} frames).")

    # 3. Load Model LiMaVLM
    print("🧠 Đang khởi tạo LiMaVLM...")
    model = LiMaVLM(d_model=256, d_text=512, num_blocks=3).to(device)
    
    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt if 'model_state' not in ckpt else ckpt['model_state'], strict=False)
        print("✅ Đã nạp trọng số thành công!")
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy {args.weights}. Chạy ngẫu nhiên!")
    model.eval()

    # Pipeline transform ảnh
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    processed_frames = []
    states = None # Bộ nhớ Mamba
    
    print("🚀 Bắt đầu Inference (Streaming Chunk)...")
    
    # 4. Vòng lặp Inference
    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        chunk_orig = orig_frames[start:end]
        
        # Tiền xử lý list numpy thành tensor cho model
        tensors = [transform(img) for img in chunk_orig]
        input_v = torch.stack(tensors, dim=1).unsqueeze(0).to(device) # [1, 3, T, H, W]

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_v, text_tokens, states=states)
            states = outputs["memory_states"]
            p_hm, p_sz, p_off = [x.float() for x in outputs["tracking_heads"]]
            
        current_chunk_len = len(chunk_orig)
        p_hm = p_hm.view(current_chunk_len, 1, p_hm.size(-2), p_hm.size(-1))
        p_sz = p_sz.view(current_chunk_len, 2, p_sz.size(-2), p_sz.size(-1))
        p_off = p_off.view(current_chunk_len, 2, p_off.size(-2), p_off.size(-1))
            
        dets_batch = decode_centernet_topk(p_hm, p_sz, p_off, k=5)
            
        for i in range(current_chunk_len):
            # Chuyển ngược RGB sang BGR để vẽ bằng OpenCV
            img_draw = cv2.cvtColor(chunk_orig[i], cv2.COLOR_RGB2BGR)
            h_orig, w_orig = img_draw.shape[:2]
            
            # --- VẼ HEATMAP SPOTLIGHT (GIỮ MÀU GỐC) ---
            hm_frame = p_hm[i, 0].cpu().numpy() 
            hm_resized = cv2.resize(hm_frame, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
            hm_resized = np.clip(hm_resized, 0, 1)
            hm_mask = np.stack([hm_resized]*3, axis=-1)
            
            base_brightness = 0.3 # Tối đi 70% ở những chỗ không có xe
            final_mask = np.clip(hm_mask + base_brightness, 0, 1)
            img_draw = (img_draw.astype(np.float32) * final_mask).astype(np.uint8)
            
            # --- VẼ PREDICTION BOXES ---
            dets = dets_batch[i]
            for d in range(5):
                score, x1, y1, x2, y2 = dets[d].numpy()
                if score > 0.05: # Threshold thấp để thấy fading
                    ix1, iy1 = int(x1 * w_orig), int(y1 * h_orig)
                    ix2, iy2 = int(x2 * w_orig), int(y2 * h_orig)
                    
                    color = (0, int(255 * score), 0) # Xanh mờ dần
                    cv2.rectangle(img_draw, (ix1, iy1), (ix2, iy2), color, 2)
                    
                    if score > 0.7: # Chỉ hiện số ở box tự tin cao
                        cv2.putText(img_draw, f"{score:.2f}", (ix1, iy1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- IN THÔNG TIN LÊN VIDEO ---
            cv2.putText(img_draw, f"Custom Text: {args.query_text}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
            processed_frames.append(img_draw)

    # 5. Lưu Video
    print(f"🎬 Đang lưu video kết quả: {args.output_video}")
    h_target, w_target = processed_frames[0].shape[:2]
    fps = 10.0 # Tùy chỉnh fps nếu video của bạn mượt hơn
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (w_target, h_target))

    for frame in processed_frames:
        out.write(frame)
    out.release()
    print("✅ Hoàn tất thành công!")

# ==========================================
# 4. CHẠY SCRIPT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiMaVLM Custom Video & Text Inference")
    parser.add_argument('--video_path', type=str, required=True, help="Đường dẫn đến file video mp4")
    parser.add_argument('--query_text', type=str, required=True, help="Câu lệnh tiếng Anh tìm xe")
    parser.add_argument('--weights', type=str, default='checkpoint_epoch_80.pth', help="Đường dẫn model weights")
    parser.add_argument('--output_video', type=str, default='custom_result.mp4', help="Tên file video đầu ra")
    parser.add_argument('--chunk_size', type=int, default=8, help="Số frames đưa vào Mamba 1 lần (ram yếu chỉnh 4)")
    
    args = parser.parse_args()
    run_custom_inference(args)