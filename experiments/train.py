import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.ops as ops # <-- THÊM THƯ VIỆN NÀY CHO GIOU LOSS

from models.fusion.lima import LiMaVLM
from experiments.utils.cityflownl import CityFlowNLDataset
from experiments.utils.loss import HierarchicalSupConLoss

def train_epoch(model, dataloader, optimizer, epoch, device, accumulation_steps=4):
    """
    Vòng lặp huấn luyện 1 Epoch tối ưu cho RTX 4060 (8GB VRAM)
    Kiến trúc End-to-End: Vừa truy xuất Text (Gate) Vừa Regression BBox
    """
    model.train()
    scaler = GradScaler()
    h_supcon_criterion = HierarchicalSupConLoss(temperature=0.07, alpha=0.5).to(device)
    
    total_loss_epoch = 0
    optimizer.zero_grad() 

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
    
    for i, batch in pbar:
        # Chuyển dữ liệu lên GPU
        video_input = batch['video'].to(device)       # [B, 3, T, 112, 112]
        text_emb = batch['text_emb'].to(device)       # [B, d_text]
        is_match = batch['is_match'].to(device)       # [B]
        target_bbox = batch['bbox'].to(device)        # [B, 4] dạng [c_x, c_y, w, h]
        coarse_labels = batch['coarse_label'].to(device) 
        fine_labels = batch['fine_label'].to(device)  

        with autocast():
            # Chạy qua mạng Li-Ma
            bbox_pred, gate_activation, features = model(video_input, text_emb)
            
            # --- TÍNH TOÁN CÁC HÀM LOSS ---
            
        # a. Loss Cổng Text (Binary Cross Entropy) - Dạy mạng biết video có chứa xe khớp text không
        loss_gate = F.binary_cross_entropy(gate_activation.float().view(-1), is_match.float().view(-1))
        
        # b. Loss Phân cấp H-SupCon 
        valid_mask = (is_match == 1.0)
        loss_h_supcon = torch.tensor(0.0).to(device)
        # Chỉ tính nếu có ít nhất 2 mẫu hợp lệ VÀ nhãn không bị gán mặc định (0) toàn bộ
        if valid_mask.sum() > 1 and coarse_labels[valid_mask].sum() > 0: 
            try:
                loss_h_supcon = h_supcon_criterion(features[valid_mask], 
                                                    coarse_labels[valid_mask], 
                                                    fine_labels[valid_mask])
            except Exception as e:
                pass # Bỏ qua nếu dữ liệu nhãn giả không đủ đa dạng để tính Contrastive
            
        # c. Loss Bounding Box (L1 + GIoU)
        # Khởi tạo danh sách chứa loss của từng sample trong batch
        batch_bbox_losses = []
        
        for b in range(video_input.size(0)):
            if is_match[b] == 1.0:
                # 1. L1 Loss
                l1 = F.l1_loss(bbox_pred[b], target_bbox[b])
                
                # 2. GIoU Loss
                pred_box_xyxy = ops.box_convert(bbox_pred[b].unsqueeze(0), in_fmt='cxcywh', out_fmt='xyxy')
                target_box_xyxy = ops.box_convert(target_bbox[b].unsqueeze(0), in_fmt='cxcywh', out_fmt='xyxy')
                giou = ops.generalized_box_iou_loss(pred_box_xyxy, target_box_xyxy).squeeze()
                
                # Lưu tổng loss của sample này (có thể thêm trọng số cho L1 và GIoU nếu muốn)
                batch_bbox_losses.append(l1 + giou)
            else:
                # Nếu text không khớp, ép mạng xuất ra Box [0,0,0,0]
                # l1_loss lúc này đóng vai trò như penalty
                l1_penalty = F.l1_loss(bbox_pred[b], torch.zeros_like(bbox_pred[b]))
                batch_bbox_losses.append(l1_penalty)
        
        # Tính trung bình toàn bộ list bằng torch.stack (rất an toàn cho đồ thị tính toán)
        if len(batch_bbox_losses) > 0:
            loss_bbox = torch.stack(batch_bbox_losses).mean()
        else:
            loss_bbox = torch.tensor(0.0, device=device, requires_grad=True)
        
        loss_bbox = loss_bbox / video_input.size(0) # Trung bình theo batch

        # d. Tổng hợp Loss
        # Hệ số Loss BBox thường cần lớn hơn (vd: 2.0 hoặc 5.0) vì giá trị L1/GIoU rất nhỏ so với BCE
        loss = 2.0 * loss_gate + 0.5 * loss_h_supcon + 5.0 * loss_bbox
        
        loss = loss / accumulation_steps

        # Backward Pass thông qua Scaler
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss_epoch += loss.item() * accumulation_steps
        pbar.set_postfix({
            'Loss': f"{(total_loss_epoch / (i + 1)):.4f}", 
            'Gate': f"{loss_gate.item():.4f}",
            'Box': f"{loss_bbox.item():.4f}"
        })
        
    return total_loss_epoch / len(dataloader)


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {DEVICE}")
    
    BATCH_SIZE = 2      # Để vừa 8GB VRAM (Mỗi batch chứa video 8 frame)
    ACCUM_STEPS = 4     # Batch size thực tế cập nhật gradient = 8
    EPOCHS = 50

    print("Đang khởi tạo Dataset...")
    train_dataset = CityFlowNLDataset(
        data_root="./data/data", # SỬA LẠI: Trỏ vào folder chứa VIDEO/FRAME TOÀN CẢNH, không dùng folder crops
        json_path='./data/data/train-tracks.json', # Tên file JSON gốc thường có đuôi -tracks
        num_frames=8,
        img_size=112,
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, # Tăng lên 4 nếu CPU của bạn có từ 8 luồng trở lên để nạp data nhanh hơn
        pin_memory=True
    )

    print("Đang khởi tạo Kiến trúc Li-Ma...")
    model = LiMaVLM(
        in_channels=3, 
        d_model=128, 
        d_text=512, 
        num_frames=8, 
        img_size=112
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)

    print("BẮT ĐẦU HUẤN LUYỆN!")
    print("===================")
    
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, epoch, DEVICE, ACCUM_STEPS)
        
        print(f"\n[Kết quả Epoch {epoch}] Trung bình Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "lima_cityflownl_detection_best.pth")
            print(">>> Đã lưu model tốt nhất!")
            
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()