import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

# Import model kiến trúc của bạn
from models.fusion.lima import LiMaVLM

def get_text_embedding(text, device):
    """Trích xuất CLIP feature"""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.eval()
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = text_encoder(**inputs).pooler_output
    return emb

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang khởi động Video Tracking trên {DEVICE}...")

    # ==========================================
    # CẤU HÌNH ĐẦU VÀO
    # ==========================================
    INPUT_FRAME_FOLDER = "./data/data/train/S01/c001/img1" 
    QUERY_TEXT = "a white suv driving straight" # Đổi thành xe bạn muốn tìm
    WEIGHT_PATH = "lima_cityflownl_detection_best.pth"
    OUTPUT_VIDEO_PATH = "output_tracked_video.mp4"
    FPS = 10 # Số khung hình/giây của video đầu ra
    NUM_FRAMES = 8
    IMG_SIZE = 112
    # ==========================================

    # 1. Tải Text Embedding
    print(f"Đang mã hóa câu lệnh: '{QUERY_TEXT}'...")
    text_emb = get_text_embedding(QUERY_TEXT, DEVICE)

    # 2. Tải Model
    print("Đang tải trọng số Li-Ma VLM...")
    model = LiMaVLM(in_channels=3, d_model=128, d_text=512, num_frames=NUM_FRAMES, img_size=IMG_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.eval()

    # 3. Chuẩn bị dữ liệu Video
    all_frames = sorted([f for f in os.listdir(INPUT_FRAME_FOLDER) if f.endswith('.jpg')])
    total_frames = len(all_frames)
    if total_frames < NUM_FRAMES:
        print("Video quá ngắn!")
        return

    # Lấy kích thước gốc từ frame đầu tiên để cấu hình VideoWriter
    first_img = cv2.imread(os.path.join(INPUT_FRAME_FOLDER, all_frames[0]))
    orig_h, orig_w, _ = first_img.shape

    # Khởi tạo VideoWriter (Dùng codec mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (orig_w, orig_h))

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Bắt đầu Tracking: Tổng cộng {total_frames - NUM_FRAMES + 1} frames...")
    
    # 4. CHẠY CỬA SỔ TRƯỢT (SLIDING WINDOW)
    # Trượt từ đầu đến cuối video, mỗi lần lấy 8 frames
    for start_idx in tqdm(range(total_frames - NUM_FRAMES + 1)):
        window_frames = all_frames[start_idx : start_idx + NUM_FRAMES]
        
        video_tensor = []
        target_frame_img = None
        
        for i, frame_name in enumerate(window_frames):
            img_path = os.path.join(INPUT_FRAME_FOLDER, frame_name)
            img = cv2.imread(img_path)
            
            # Lưu lại frame ở giữa để vẽ Box (Frame thứ 4, index = 3 hoặc 4)
            if i == NUM_FRAMES // 2:
                target_frame_img = img.copy()

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tensor = transform(img_pil)
            video_tensor.append(img_tensor)

        # Đẩy vào Model
        video_tensor = torch.stack(video_tensor, dim=1).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            bbox_pred, gate_act, _ = model(video_tensor, text_emb)

        bbox = bbox_pred.squeeze().cpu().numpy()
        confidence = gate_act.item()

        # 5. VẼ BOX NẾU ĐỘ TỰ TIN CAO (Ngưỡng > 60%)
        if confidence > 0.00:
            c_x, c_y, norm_w, norm_h = bbox
            w = int(norm_w * orig_w)
            h = int(norm_h * orig_h)
            x_center = int(c_x * orig_w)
            y_center = int(c_y * orig_h)

            x1 = max(0, int(x_center - w / 2))
            y1 = max(0, int(y_center - h / 2))
            x2 = min(orig_w, int(x_center + w / 2))
            y2 = min(orig_h, int(y_center + h / 2))

            # Vẽ Hộp màu Xanh lá
            cv2.rectangle(target_frame_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Vẽ thanh nhãn
            label = f"Match: {confidence*100:.0f}%"
            cv2.rectangle(target_frame_img, (x1, y1 - 30), (x1 + 180, y1), (0, 255, 0), -1)
            cv2.putText(target_frame_img, label, (x1 + 5, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            # Nếu không tự tin (xe bị che khuất hoặc text sai), hiển thị cảnh báo đỏ
            cv2.putText(target_frame_img, "No Match / Occluded", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 6. Ghi frame đã vẽ vào Video Output
        out_video.write(target_frame_img)

    # Đóng luồng ghi video
    out_video.release()
    print(f"\n✅ Hoàn tất! Video Tracking đã được lưu tại: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()