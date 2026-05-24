import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import math

# ==========================================
# 0. IMPORT MODULES CỦA BẠN
# ==========================================
from experiments.utils.dataset import CityFlowNLDataset
from models.lima import LiMaVLM 

# ==========================================
# 1. INFO-NCE VỚI CROSS-BATCH MEMORY (XBM)
# ==========================================
class XBMInfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v_emb, t_emb, queue_v, queue_t, logit_scale):
        """
        v_emb, t_emb: Đặc trưng của batch hiện tại [Batch, D]
        queue_v, queue_t: Đặc trưng lưu trong hàng đợi [D, Queue_Size]
        """
        batch_size = v_emb.shape[0]
        
        # Mở rộng Key bằng cách nối current batch với Queue
        extended_v = torch.cat([v_emb, queue_v.T], dim=0) # [Batch + Queue, D]
        extended_t = torch.cat([t_emb, queue_t.T], dim=0) # [Batch + Queue, D]
        
        # Tính Logits: [Batch, Batch + Queue_Size]
        sim_i2t = torch.matmul(v_emb, extended_t.T) * logit_scale
        sim_t2i = torch.matmul(t_emb, extended_v.T) * logit_scale
        
        # Ground Truth luôn nằm trên đường chéo của ma trận [Batch, Batch]
        labels = torch.arange(batch_size, device=v_emb.device)
        
        loss_i2t = F.cross_entropy(sim_i2t, labels)
        loss_t2i = F.cross_entropy(sim_t2i, labels)
        
        return (loss_i2t + loss_t2i) / 2.0

class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=80):
        super().__init__()
        self.m, self.gamma = m, gamma
        self.O_p, self.O_n = 1 + m, -m
        self.Delta_p, self.Delta_n = 1 - m, m

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        sim_mat = torch.matmul(features, features.t())
        loss, valid_pairs = 0.0, 0
        batch_size = features.size(0)
        for i in range(batch_size):
            pos_mask = labels == labels[i]
            pos_mask[i] = False 
            neg_mask = labels != labels[i]
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                s_p = sim_mat[i][pos_mask]
                s_n = sim_mat[i][neg_mask]
                alpha_p = torch.relu(self.O_p - s_p.detach())
                alpha_n = torch.relu(s_n.detach() - self.O_n)
                logit_p = -self.gamma * alpha_p * (s_p - self.Delta_p)
                logit_n = self.gamma * alpha_n * (s_n - self.Delta_n)
                loss_i = torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)
                loss += F.softplus(loss_i) 
                valid_pairs += 1
        if valid_pairs > 0:
            return loss / valid_pairs
        return torch.tensor(0.0, requires_grad=True, device=features.device)

# ==========================================
# 2. VÒNG LẶP HUẤN LUYỆN 1 EPOCH
# ==========================================
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, loss_list, device, epoch, accumulation_steps=4):
    model.train()
    total_epoch_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        # Lấy dữ liệu
        crop_frames = batch["crop"].to(device, non_blocking=True)
        original_frames = batch["frame"].to(device, non_blocking=True)
        
        color_emb = batch["color_embedding"].to(device, non_blocking=True)
        type_emb = batch["type_embedding"].to(device, non_blocking=True)
        motion_emb = batch["motion_embedding"].to(device, non_blocking=True)
        lang_embeds_raw = batch["text_embeds"].to(device, non_blocking=True) 
        id_car = batch["car_id"].to(device, non_blocking=True)

        with autocast():
            # 1. Trích xuất nhánh ảnh (Visual Embeds)
            outputs = model.encode_image(crop_frames, original_frames)
            visual_embeds = outputs["visual_embeds"] 
            cls_logit = outputs["id_logits"]
            
            # 2. Hàm hỗ trợ xử lý nhánh Text
            def process_text_emb(raw_emb):
                if raw_emb.dim() == 3: raw_emb = raw_emb.mean(dim=1)
                return model.encode_text(raw_emb)

            proj_color = process_text_emb(color_emb)
            proj_type = process_text_emb(type_emb)
            proj_motion = process_text_emb(motion_emb)
            proj_lang = process_text_emb(lang_embeds_raw)

            # 3. Lấy Vector từ Queue và Logit Scale
            queue_v = model.image_queue.clone().detach()
            queue_t = model.text_queue.clone().detach()
            logit_scale = model.logit_scale.exp()

            # 4. Tính Multi-Granularity InfoNCE
            def calc_xbm_infonce(v_emb, t_emb):
                return loss_list['infoNCE'](v_emb, t_emb, queue_v, queue_t, logit_scale)

            loss_nce_color = calc_xbm_infonce(visual_embeds, proj_color)
            loss_nce_type = calc_xbm_infonce(visual_embeds, proj_type)
            loss_nce_motion = calc_xbm_infonce(visual_embeds, proj_motion)
            loss_nce_lang = calc_xbm_infonce(visual_embeds, proj_lang)
            
            loss_infoNCE = (loss_nce_color + loss_nce_type + loss_nce_motion + loss_nce_lang) / 4.0

            # 5. Tính Circle Loss và Re-ID Loss
            pair_features = torch.cat([visual_embeds, proj_lang], dim=0)
            pair_labels = torch.cat([id_car, id_car], dim=0).long()
            loss_circle = loss_list['CircleLoss'](pair_features, pair_labels)
            loss_ce = F.cross_entropy(cls_logit, id_car.long())
            
            loss_total = (2.0*loss_infoNCE + 0.2*loss_circle + 0.5 * loss_ce) / accumulation_steps
        
        # 6. Backward
        scaler.scale(loss_total).backward()
        
        # 7. Optimizer Step & Queue Update
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            # 🔥 CẬP NHẬT QUEUE (Chỉ đưa đặc trưng vào queue sau khi kết thúc Gradient Accumulation)
            model._dequeue_and_enqueue(visual_embeds.detach(), proj_lang.detach())
        
        # 8. Logging
        real_loss_value = loss_total.item() * accumulation_steps
        total_epoch_loss += real_loss_value
        
        pbar.set_postfix({
            "Loss": f"{real_loss_value:.3f}",
            "NCE_XBM": f"{loss_infoNCE.item():.3f}",
            "Cir": f"{loss_circle.item():.3f}",
            "CE": f"{loss_ce.item():.3f}"
        })
        
    return total_epoch_loss / len(dataloader)

# ==========================================
# 3. SETUP OPTIMIZER
# ==========================================
def build_optimizer_and_scheduler(model, total_steps, base_lr=1e-4):
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        # Áp dụng Learning Rate nhỏ hơn cho Backbone để tránh phá vỡ pre-trained weights
        if "video_backbone" in name or "text_proj" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': base_lr * 0.1}, 
        {'params': head_params, 'lr': base_lr}            
    ], weight_decay=1e-4)

    scheduler = OneCycleLR(
        optimizer, max_lr=[base_lr * 0.1, base_lr], 
        total_steps=total_steps, pct_start=0.1, anneal_strategy='cos'
    )
    return optimizer, scheduler

# ==========================================
# 4. HÀM MAIN
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Bắt đầu quá trình huấn luyện trên {device.upper()}...")

    # --- CẤU HÌNH CƠ BẢN ---
    num_epochs = 300
    batch_size = 32
    accumulation_steps = 4  
    queue_size = 1024        # Kích thước Batch ảo
    save_dir = "./checkpoints/models"
    os.makedirs(save_dir, exist_ok=True)

    class DataCfg:
        ROOT_DIR = "./"
        DATA_DIR = "./data"
        CITYFLOW_PATH = "cityflownl/data"
        CROP_AREA = 1.6666667
        def clone(self): return self
    data_cfg = DataCfg()

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- TẢI DATASET ---
    print("📥 Đang tải Dataset...")
    train_dataset = CityFlowNLDataset(
        data_cfg=data_cfg, json_path="data/json/train_clean.json",
        text_emb_path="data/data/clip_text_tokens_extracted.pt",
        transform=train_transform, Random=True, type="train"
    )
    
    num_cars_in_dataset = len(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )

    # --- KHỞI TẠO MÔ HÌNH ---
    print(f"🧠 Khởi tạo LiMaVLM với số ID xe: {num_cars_in_dataset} và Queue Size: {queue_size}...")
    model = LiMaVLM(
        d_model=256, d_text_in=512, num_blocks=4, 
        num_classes=num_cars_in_dataset, queue_size=queue_size
    ).to(device)

    # --- SETUP OPTIMIZER & LOSS ---
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    
    optimizer, scheduler = build_optimizer_and_scheduler(model, total_steps, base_lr=1e-4)
    scaler = GradScaler()
    
    loss_list = {
        'infoNCE': XBMInfoNCELoss().to(device),
        'CircleLoss': CircleLoss(m=0.25, gamma=80).to(device)
    }

    # --- BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN ---
    print("\n🔥 Bắt đầu Training...")
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, 
            loss_list, device, epoch, accumulation_steps
        )
        print(f"📊 Epoch {epoch} hoàn thành | Trung bình Loss: {epoch_loss:.4f}\n")
        
        # 1. Lưu Best Model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(save_dir, "limavlm_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"⭐ Đã cập nhật và lưu Best Model (Loss: {best_loss:.4f}) tại {best_path}")
            
        # 2. Lưu Checkpoint định kỳ mỗi 5 epoch
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f"limavlm_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"📦 Đã lưu Checkpoint định kỳ tại {checkpoint_path}")

if __name__ == "__main__":
    main()