import torch
import numpy as np
import cv2
import os
import torchvision.ops as ops

def visualize_parts(model_path, dataset, device, output_dir="./vis_results", num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurations
    FEAT_STRIDE = 16
    IMAGE_CROP_SIZE = 336  # Kích thước cắt xe đưa vào Backbone 
    FEATURE_ROI_SIZE = 7   # Kích thước grid_sample bên trong model
    # ---------------------------------------------

    # 1. Load model
    checkpoint = torch.load(model_path, map_location=device)
    from models.microlocal import MicroLocalization 
    model = MicroLocalization(embed_dim=512, num_parts=5, roi_size=FEATURE_ROI_SIZE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 2. Lấy mẫu ngẫu nhiên từ dataset
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        data = dataset[idx]
        track_id = data['track_id']
        
        # Thêm chiều Batch (B=1) cho tensor
        video = data['video'].unsqueeze(0).to(device)      # [1, T, 3, H, W]
        bboxes = data['bbox'].unsqueeze(0).to(device)      # [1, T, 4]
        text_emb = data['vehicle_motion_text_emb'].unsqueeze(0).to(device) # [1, L, D]
        
        # Chỉ lấy Frame đầu tiên (t=0) để visualize cho gọn
        frame_idx = 0
        img_tensor = video[:, frame_idx, :, :, :] # [1, 3, 384, 384]
        box_tensor = bboxes[:, frame_idx, :]      # [1, 4] (cx, cy, w, h)
        
        # =====================================================================
        # BƯỚC QUAN TRỌNG: XỬ LÝ BBOX VÀ CẮT ROI 
        # =====================================================================
        # 1. Scale box ngược lại kích thước ảnh gốc (x16)
        box_scaled = box_tensor * FEAT_STRIDE
        
        # 2. Chuyển từ [cx, cy, w, h] sang [x1, y1, x2, y2]
        cx, cy, w, h = box_scaled[:, 0], box_scaled[:, 1], box_scaled[:, 2], box_scaled[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # 3. Format box cho roi_align: [batch_idx, x1, y1, x2, y2]
        batch_indices = torch.zeros((1, 1), device=device, dtype=torch.float32)
        rois_boxes = torch.cat([batch_indices, x1.unsqueeze(1), y1.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)], dim=1)
        
        # 4. Thực hiện cắt ảnh bằng CUDA roi_align
        # Đầu ra là ảnh chiếc xe đã được crop & resize chuẩn [1, 3, 112, 112]
        vehicle_roi = ops.roi_align(img_tensor, rois_boxes, output_size=IMAGE_CROP_SIZE, spatial_scale=1.0, aligned=True)
        
        # =====================================================================
        # BƯỚC FORWARD PASS
        # =====================================================================
        with torch.no_grad():
            out, coords, visibility, attn, scores = model(vehicle_roi, text_emb)

        # =====================================================================
        # VẼ HÌNH (VẼ LÊN CHÍNH ẢNH XE ĐÃ CẮT - CỰC KỲ TRỰC QUAN)
        # =====================================================================
        # 1. Lấy ảnh xe ra khỏi GPU và Unnormalize
        img_vis = vehicle_roi.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_vis = (img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) 
        img_vis = (np.clip(img_vis, 0, 1) * 255).astype(np.uint8)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR) # OpenCV dùng BGR
        
        H, W, _ = img_vis.shape # Kích thước lúc này là 112x112 (hoặc tùy cấu hình của bạn)

        # 2. Lấy tọa độ và điểm số
        part_coords = coords[0].cpu().numpy() # [5, 2] -> (cx, cy) dải 0-1
        part_scores = scores[0].cpu().numpy() # [5, 1]
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)] # Các màu sắc khác nhau
        
        # Resize ảnh lên to một chút để dễ nhìn (ví dụ 300x300)
        zoom_factor = 3
        img_vis = cv2.resize(img_vis, (W * zoom_factor, H * zoom_factor), interpolation=cv2.INTER_NEAREST)
        H_zoom, W_zoom = img_vis.shape[:2]

        # 3. Vẽ lên ảnh
        for i in range(len(part_coords)):
            cx_norm, cy_norm = part_coords[i]
            
            # Map tọa độ 0-1 vào ảnh đã zoom
            px = int(cx_norm * W_zoom)
            py = int(cy_norm * H_zoom)
            score = part_scores[i][0]
            
            # Vẽ 1 dấu '+' tại tâm thay vì chấm tròn che mất chi tiết
            thickness = 2
            size = 8
            cv2.line(img_vis, (px - size, py), (px + size, py), colors[i], thickness)
            cv2.line(img_vis, (px, py - size), (px, py + size), colors[i], thickness)
            
            # Vẽ vòng tròn nhạt mô phỏng "vùng attention" (roi_size hạt mịn)
            cv2.circle(img_vis, (px, py), int(W_zoom * 0.15), colors[i], 1)
            
            # Ghi text (Part Index + Score)
            cv2.putText(img_vis, f"P{i}:{score:.2f}", (px + 10, py - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1, cv2.LINE_AA)

        # 4. Lưu kết quả
        save_path = os.path.join(output_dir, f"vis_{track_id}.jpg")
        cv2.imwrite(save_path, img_vis)
        print(f" Đã lưu ảnh: {save_path}")
        
if __name__ == "__main__":
    # Ví dụ sử dụng
    model_checkpoint = "./checkpoints/micro_localization/best_model_epoch_59.pth"
    from experiments.utils.dataset import CityFlowNLDataset
    dataset = CityFlowNLDataset(data_root="./data/data", json_path="./data/data/train-tracks.json", text_emb_path="./data/data/clip_text_tokens_extracted.pt", img_size=512)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize_parts(model_checkpoint, dataset, device)