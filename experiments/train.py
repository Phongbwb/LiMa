import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import json

# Tạm giả định bạn đã lưu các class này trong các file tương ứng
# Nếu bạn gom chung vào 1 file thì không cần import
from models.lima import LiMaVLM
from experiments.utils.dataset import CityFlowNLDataset

# ==========================================
# 1. CÁC HÀM LOSS CHUYÊN DỤNG
# ==========================================
def focal_loss_centernet(pred_hm, gt_hm, alpha=2, beta=4):
    """Focal Loss cho Heatmap"""
    pred_hm = torch.clamp(pred_hm, min=1e-4, max=1 - 1e-4)
    pos_inds = gt_hm.eq(1).float()
    neg_inds = gt_hm.lt(1).float()

    neg_weights = torch.pow(1 - gt_hm, beta)
    
    pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, alpha) * pos_inds
    neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos

def reg_l1_loss(pred, gt, mask):
    """L1 Loss có mặt nạ (Chỉ phạt tại vị trí có xe)"""
    pred = pred * mask
    gt = gt * mask
    loss = F.l1_loss(pred, gt, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

def in_batch_contrastive_loss(video_feat, text_feat, temperature=0.07):
    """InfoNCE Loss cho Video-Text Retrieval"""
    logits = (video_feat @ text_feat.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.T, labels)
    return (loss_v2t + loss_t2v) / 2

# ==========================================
# 2. HÀM VALIDATION
# ==========================================
@torch.no_grad()
def validate_limavlm(model, val_loader, queries_json_path, text_embs_dict, device='cuda'):
    model.eval()
    
    # --- BƯỚC MỚI: Đọc file JSON từ đầu để map track_id -> query ---
    print("📖 Đang nạp metadata để chuẩn bị text cho Gallery...")
    with open(val_loader.dataset.json_path, 'r') as f:
        full_track_ids = list(json.load(f).keys())
        
    with open(queries_json_path, 'r') as f:
        queries_data = json.load(f)
    full_query_ids = list(queries_data.keys())
    # -------------------------------------------------------------

    # 1. Trích xuất đặc trưng Gallery
    gallery_feats_dict = {}
    print("📹 Đang trích xuất Gallery features (Sử dụng Real Text)...")
    for batch in val_loader:
        video = batch['video'].to(device)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        ids = batch['track_id']
        
        # --- BẮT ĐẦU THAY THẾ DUMMY TEXT ---
        batch_text_tokens = []
        for tid in ids:
            original_idx = full_track_ids.index(tid)
            q_id = full_query_ids[original_idx]
            nl_text = queries_data[q_id]['nl'][0].strip().lower()
            
            # Lấy text embedding tương ứng (shape: [32, 512])
            t_feat = text_embs_dict.get(nl_text, torch.zeros(32, 512))
            batch_text_tokens.append(t_feat)
            
        # Ghép thành batch [B, 32, 512] và đưa lên GPU
        real_text_tokens = torch.stack(batch_text_tokens).to(device)
        # --- KẾT THÚC THAY THẾ DUMMY TEXT ---
        
        with autocast():
            # Truyền text thật của track_id đó vào mô hình
            outputs = model(video, real_text_tokens) 
            
        feats = outputs["retrieval_feat"].cpu()
        for i, tid in enumerate(ids):
            gallery_feats_dict[tid] = feats[i]

    ordered_track_ids = val_loader.dataset.track_ids 
    gallery_feats = torch.stack([gallery_feats_dict[tid] for tid in ordered_track_ids])

    # --- BẮT ĐẦU FIX LỖI LOGIC SO KHỚP (Giữ nguyên 100% logic của bạn) ---
    print("📖 Đang nạp Query features (Khớp vị trí gốc)...")
    
    query_feats = []
    
    # Duyệt qua từng video S01 đã được lọc
    for tid in ordered_track_ids:
        # Tìm vị trí (index) gốc của video này trong file test-tracks
        original_idx = full_track_ids.index(tid)
        
        # Bốc đúng query ở vị trí tương ứng trong file test-queries
        q_id = full_query_ids[original_idx]
        nl_text = queries_data[q_id]['nl'][0].strip().lower()
        
        # Tra cứu embedding
        t_feat = text_embs_dict.get(nl_text, torch.zeros(32, 512)).mean(dim=0)
        query_feats.append(F.normalize(t_feat, dim=0))
    
    query_feats = torch.stack(query_feats) # [num_available, 512]
    # --- KẾT THÚC FIX LỖI LOGIC ---

    # 3. Tính Recall@1 (Lúc này index i chắc chắn khớp với index i)
    sim_matrix = query_feats @ gallery_feats.T
    targets = torch.arange(len(ordered_track_ids))
    preds = sim_matrix.argmax(dim=1)
    
    recall_1 = (preds == targets).float().mean().item()
    print(f"📊 Kết quả Validation (S01): Recall@1 = {recall_1:.4f}")
    
    return recall_1

# ==========================================
# 3. VÒNG LẶP HUẤN LUYỆN CHÍNH
# ==========================================
def train_limavlm(model, train_loader, val_loader, epochs=15, val_interval=2, device='cuda'):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    model.to(device)
    best_recall = 0.0
    text_embs_dict = torch.load("./data/data/clip_text_tokens.pt")
    for epoch in range(1, epochs + 1):
        # --- CURRICULUM LEARNING ---
        if epoch == 1:
            print("\n🚀 STAGE 1: Warm-up Macro-Tracking (Epoch 1-5)")
            for param in model.blocks.parameters(): param.requires_grad = False
            for param in model.micro_loc.parameters(): param.requires_grad = False
            for param in model.liquid_memory.parameters(): param.requires_grad = False
            
            lambda_hm, lambda_sz, lambda_off = 1.0, 0.1, 1.0
            lambda_div, lambda_ret = 0.0, 0.0 
            
        elif epoch == 20:
            print("\n🚀 STAGE 2: Full Multi-Task Learning (Epoch 6-15)")
            for param in model.parameters(): param.requires_grad = True
            
            lambda_hm, lambda_sz, lambda_off = 1.0, 0.1, 1.0
            lambda_div, lambda_ret = 0.5, 5.0 

        model.train()
        total_loss_epoch = 0

        for batch_idx, batch in enumerate(train_loader):
            video = batch['video'].to(device)       # [B, T, 3, H, W]
            gt_hm = batch['hm'].to(device)          # [B, T, 1, H', W']
            gt_sz = batch['sz'].to(device)          # [B, T, 2, H', W']
            gt_off = batch['off'].to(device)        # [B, T, 2, H', W']
            text_tokens = batch['text_tokens'].to(device) # [B, 32, 512]
            
            B, T = video.shape[:2]
            video = video.permute(0, 2, 1, 3, 4).contiguous()
            
            # Flatten trục thời gian cho CenterNet
            gt_hm_2d = gt_hm.view(B*T, 1, gt_hm.size(-2), gt_hm.size(-1))
            gt_sz_2d = gt_sz.view(B*T, 2, gt_sz.size(-2), gt_sz.size(-1))
            gt_off_2d = gt_off.view(B*T, 2, gt_off.size(-2), gt_off.size(-1))
            mask = gt_hm_2d.eq(1).float().expand_as(gt_sz_2d)

            optimizer.zero_grad()

            # Forward Pass với AMP
            with autocast():
                outputs = model(video, text_tokens)
                
                pred_hm, pred_sz, pred_off = outputs["tracking_heads"]
                retrieval_feat = outputs["retrieval_feat"]
                coords, vis = outputs["micro_parts"]
                
                # Tính Losses
                l_hm = focal_loss_centernet(pred_hm, gt_hm_2d)
                l_sz = reg_l1_loss(pred_sz, gt_sz_2d, mask)
                l_off = reg_l1_loss(pred_off, gt_off_2d, mask)
                
                l_div = model.micro_loc.get_diversity_loss(coords)
                
                text_global = F.normalize(text_tokens.mean(dim=1), dim=-1)
                l_ret = in_batch_contrastive_loss(retrieval_feat, text_global)

                loss = (lambda_hm * l_hm) + (lambda_sz * l_sz) + \
                       (lambda_off * l_off) + (lambda_div * l_div) + \
                       (lambda_ret * l_ret)

            # Backward Pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss_epoch += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (HM: {l_hm.item():.4f}, Ret: {l_ret.item():.4f})")

        scheduler.step()
        avg_loss = total_loss_epoch / len(train_loader)
        print(f"✅ Epoch {epoch} Xong! Average Loss: {avg_loss:.4f}")
        
        # --- THỰC HIỆN VALIDATION ---
        if epoch % val_interval == 0:
            current_recall = validate_limavlm(model, val_loader, "./data/data/test-queries.json", text_embs_dict, device)
            
            if current_recall > best_recall:
                best_recall = current_recall
                torch.save(model.state_dict(), "best_limavlm.pth")
                print(f"🌟 LƯU BEST MODEL MỚI (Recall@1: {best_recall:.4f})")
        
        # Lưu checkpoint dự phòng mỗi epoch
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

# ==========================================
# 4. CHẠY CHƯƠNG TRÌNH (MAIN)
# ==========================================
if __name__ == "__main__":
    # KIỂM TRA GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Đang sử dụng thiết bị: {device}")
    if device == "cuda":
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")

    # ĐƯỜNG DẪN DỮ LIỆU (BẠN CHỈNH SỬA Ở ĐÂY)
    DATA_ROOT = "./data/data"
    TRAIN_JSON = os.path.join(DATA_ROOT, "train-tracks.json")
    VAL_JSON = os.path.join(DATA_ROOT, "test-tracks.json") # Đổi thành file validation của bạn
    TEXT_EMB_PATH = os.path.join(DATA_ROOT, "clip_text_tokens.pt")

    # KHỞI TẠO DATASET VÀ DATALOADER
    print("📦 Đang chuẩn bị dữ liệu...")
    train_dataset = CityFlowNLDataset(TRAIN_JSON, DATA_ROOT, TEXT_EMB_PATH, max_frames=8)
    # Validation nên lấy max_frames cố định để đánh giá công bằng
    val_dataset = CityFlowNLDataset(VAL_JSON, DATA_ROOT, TEXT_EMB_PATH, max_frames=8) 

    # Batch_size=4 cho RTX 4060 8GB. Nếu OOM, giảm xuống 2.
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # KHỞI TẠO MÔ HÌNH
    print("🧠 Đang khởi tạo LiMaVLM...")
    model = LiMaVLM(d_model=256, d_text=512, num_blocks=3)

    # BẮT ĐẦU HUẤN LUYỆN
    print("🔥 Bắt đầu quá trình huấn luyện!")
    train_limavlm(model, train_loader, val_loader, epochs=80, val_interval=2, device=device)