import os
import json
import cv2
import numpy as np
import torch
from torchvision.transforms import v2
from tqdm import tqdm
import concurrent.futures

from models.trajectory_classifier import VehicleActionFusionModel

# ==========================================
# 1. HÀM TRÍCH XUẤT ĐẶC TRƯNG ĐỘNG HỌC 9D
# ==========================================
def compute_9d_kinematics(bboxes, img_w=1920, img_h=1080):
    T = bboxes.shape[0]
    features = torch.zeros((T, 9), dtype=torch.float32)
    eps = 1e-6 
    
    x_min, y_min, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x_c, y_c = x_min + w / 2, y_min + h / 2
    
    features[:, 0], features[:, 1] = x_c / img_w, y_c / img_h
    features[:, 2], features[:, 3] = w / img_w, h / img_h
    features[:, 4] = w / (h + eps)
    
    v_x, v_y = torch.zeros(T), torch.zeros(T)
    v_x[1:] = (x_c[1:] - x_c[:-1]) / (w[1:] + eps)
    v_y[1:] = (y_c[1:] - y_c[:-1]) / (h[1:] + eps)
    if T > 1: v_x[0], v_y[0] = v_x[1], v_y[1]
        
    features[:, 5], features[:, 6] = v_x, v_y
    
    a_x, a_y = torch.zeros(T), torch.zeros(T)
    a_x[1:] = v_x[1:] - v_x[:-1]
    a_y[1:] = v_y[1:] - v_y[:-1]
    if T > 1: a_x[0], a_y[0] = a_x[1], a_y[1]
        
    features[:, 7], features[:, 8] = a_x, a_y
    return features

# ==========================================
# 2. HÀM ĐỌC ẢNH (Bảo vệ RAM)
# ==========================================
def load_and_crop_window(args):
    """
    Thay vì tải toàn bộ Track, hàm này chỉ tải ĐÚNG 1 Cửa Sổ (6 frames).
    Xử lý xong cửa sổ nào, RAM sẽ tự động giải phóng bộ nhớ của cửa sổ đó.
    """
    window_frames, window_boxes, base_dir, transform = args
    window_imgs = []
    
    for frame_path, box in zip(window_frames, window_boxes):
        clean_path = os.path.normpath(frame_path).replace('\\', '/').lstrip('./').lstrip('/')
        real_path = os.path.join(base_dir, clean_path)
        
        img = cv2.imread(real_path)
        if img is None:
            crop_img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_min, y_min, w, h = [int(v) for v in box]
            crop_img = img[max(0, y_min):y_min+h, max(0, x_min):x_min+w]
            if crop_img.size == 0:
                crop_img = np.zeros((224, 224, 3), dtype=np.uint8)
                
        window_imgs.append(transform(crop_img))
        
    return torch.stack(window_imgs)

# ==========================================
# 3. HÀM CHÍNH - INFERENCE PIPELINE
# ==========================================
def main():
    # --- CẤU HÌNH ---
    TEST_JSON_PATH = "./data/json/test-tracks.json"   
    OUTPUT_JSON_PATH = "./submission.json"            
    CHECKPOINT_PATH = "./checkpoints/fusion_model_cfc_epoch_20.pth"
    BASE_IMG_DIR = "./data/cityflownl/data" 
    
    SEQ_LEN = 6         
    STRIDE = 2          
    IMG_W = 1920
    IMG_H = 1080
    NUM_WORKERS = 4     # Giảm số luồng xuống 4 để tránh CPU spam RAM quá lố
    
    # THÊM: Giới hạn Batch Size khi đẩy qua GPU để chống nổ VRAM
    MAX_BATCH_SIZE = 32 
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Khởi chạy quá trình Inference trên thiết bị: {DEVICE}")

    # --- KHỞI TẠO PIPELINE ẢNH ---
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- KHỞI TẠO MÔ HÌNH ---
    model = VehicleActionFusionModel(num_classes=4).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"[*] Đã nạp thành công trọng số từ: {CHECKPOINT_PATH}")
    else:
        print(f"[!] LỖI: Không tìm thấy file checkpoint tại {CHECKPOINT_PATH}")
        return
    model.eval()

    # --- ĐỌC DỮ LIỆU ĐẦU VÀO ---
    with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"[*] Đã nạp {len(test_data)} tracks từ file test.")

    results = {}

    # --- CHẠY DỰ ĐOÁN SIÊU TỐC VÀ AN TOÀN ---
    with torch.no_grad():
        for uuid, track_info in tqdm(test_data.items(), desc="Đang dự đoán", unit="track"):
            frames = track_info.get("frames", [])
            boxes = track_info.get("boxes", [])
            
            num_frames = len(frames)
            if num_frames < 2:
                results[uuid] = {"id": 0}
                continue

            # 1. Đệm dữ liệu (Pre-padding)
            if num_frames < SEQ_LEN:
                pad_len = SEQ_LEN - num_frames
                padded_frames = [frames[0]] * pad_len + frames
                padded_boxes = [boxes[0]] * pad_len + boxes
            else:
                padded_frames = frames
                padded_boxes = boxes
                
            total_frames = len(padded_frames)

            # 2. Xây dựng danh sách các Cửa Sổ Trượt (chỉ lưu Text, không tốn RAM)
            windows_info = []
            for start_idx in range(0, total_frames - SEQ_LEN + 1, STRIDE):
                windows_info.append({
                    "frames": padded_frames[start_idx : start_idx + SEQ_LEN],
                    "boxes": padded_boxes[start_idx : start_idx + SEQ_LEN]
                })
                
            # Cửa sổ cuối
            if (total_frames - SEQ_LEN) % STRIDE != 0:
                windows_info.append({
                    "frames": padded_frames[-SEQ_LEN:],
                    "boxes": padded_boxes[-SEQ_LEN:]
                })

            total_windows = len(windows_info)
            track_probs = [] # Lưu xác suất dự đoán của tất cả cửa sổ trong track này

            # 3. KỸ THUẬT CHUNKING: Băm track dài thành nhiều Mini-batches (vd: 32 cửa sổ/lần)
            for batch_start in range(0, total_windows, MAX_BATCH_SIZE):
                batch_end = min(batch_start + MAX_BATCH_SIZE, total_windows)
                mini_batch_windows = windows_info[batch_start:batch_end]
                
                # Chuẩn bị dữ liệu đa luồng (Chỉ đọc ảnh cho 32 cửa sổ này)
                thread_args = [(w["frames"], w["boxes"], BASE_IMG_DIR, val_transform) for w in mini_batch_windows]
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    # image_batch shape: [Batch, 6, 3, 224, 224]
                    image_batch = torch.stack(list(executor.map(load_and_crop_window, thread_args)))
                
                # Chuẩn bị Động học 9D
                kin_batch = []
                for w in mini_batch_windows:
                    boxes_tensor = torch.tensor(w["boxes"], dtype=torch.float32)
                    kin_batch.append(compute_9d_kinematics(boxes_tensor, img_w=IMG_W, img_h=IMG_H))
                kin_batch = torch.stack(kin_batch)
                
                # Đẩy Mini-batch lên GPU
                image_inputs = image_batch.to(DEVICE, non_blocking=True)
                kinematic_inputs = kin_batch.to(DEVICE, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(image_inputs, kinematic_inputs) 
                
                # Lấy xác suất softmax và đẩy về CPU ngay lập tức để giải phóng VRAM
                probs = torch.softmax(outputs, dim=1).cpu() 
                track_probs.append(probs)
                
                # Xóa bộ nhớ cache GPU của vòng lặp này
                del image_inputs, kinematic_inputs, outputs, probs
            
            # 4. Soft Voting tổng kết Track
            all_track_probs = torch.cat(track_probs, dim=0) # Ghép tất cả mini-batch lại
            mean_probs = torch.mean(all_track_probs, dim=0)           
            best_id = torch.argmax(mean_probs).item()       

            results[uuid] = {"id": best_id}

    # --- XUẤT FILE KẾT QUẢ ---
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[*] Hoàn tất! Đã lưu kết quả dự đoán ra file: {OUTPUT_JSON_PATH}")

if __name__ == '__main__':
    main()