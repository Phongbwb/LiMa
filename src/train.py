import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
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

    def forward(self, v_emb, t_emb, queue_v=None, queue_t=None, logit_scale=1.0):
        batch_size = v_emb.shape[0]
        
        if queue_v is not None and queue_t is not None:
            extended_v = torch.cat([v_emb, queue_v.T], dim=0)
            extended_t = torch.cat([t_emb, queue_t.T], dim=0)
        else:
            extended_v = v_emb
            extended_t = t_emb
        
        sim_i2t = torch.matmul(v_emb, extended_t.T) * logit_scale
        sim_t2i = torch.matmul(t_emb, extended_v.T) * logit_scale
        
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
# 2. VÒNG LẶP HUẤN LUYỆN & ĐÁNH GIÁ
# ==========================================
def process_batch_loss(model, batch, device, loss_list, update_queue=False):
    crop_frames = batch["crop"].to(device, non_blocking=True)
    original_frames = batch["frame"].to(device, non_blocking=True)
    
    #  [ĐÃ SỬA] Uncomment để lấy bbox_features từ dataset
    bbox_features = batch["bbox_features"].to(device, non_blocking=True)
    
    color_emb = batch["color_embedding"].to(device, non_blocking=True)
    type_emb = batch["type_embedding"].to(device, non_blocking=True)
    motion_emb = batch["motion_embedding"].to(device, non_blocking=True)
    context_emb_text = batch["context_embedding"].to(device, non_blocking=True)
    lang_embeds_raw = batch["text_embeds"].to(device, non_blocking=True) 
    id_car = batch["car_id"].to(device, non_blocking=True)

    #  [ĐÃ SỬA] Truyền thêm tham số bbox_features vào hàm encode_image
    outputs = model.encode_image(crop_frames, original_frames, bbox_features)
    
    visual_embeds = outputs["visual_embeds"] 
    visual_context = outputs.get("context_embeds", None)
    cls_logit = outputs["id_logits"]
    
    def process_text_emb(raw_emb):
        if raw_emb.dim() == 3: raw_emb = raw_emb.mean(dim=1)
        return model.encode_text(raw_emb)

    proj_color = process_text_emb(color_emb)
    proj_type = process_text_emb(type_emb)
    proj_motion = process_text_emb(motion_emb)
    proj_context_text = process_text_emb(context_emb_text)
    proj_lang = process_text_emb(lang_embeds_raw)

    queue_v = model.image_queue.clone().detach() if update_queue else None
    queue_t = model.text_queue.clone().detach() if update_queue else None
    logit_scale = model.logit_scale.exp()

    loss_nce_color = loss_list['infoNCE'](visual_embeds, proj_color, queue_v, queue_t, logit_scale)
    loss_nce_type = loss_list['infoNCE'](visual_embeds, proj_type, queue_v, queue_t, logit_scale)
    loss_nce_motion = loss_list['infoNCE'](visual_embeds, proj_motion, queue_v, queue_t, logit_scale)
    loss_nce_lang = loss_list['infoNCE'](visual_embeds, proj_lang, queue_v, queue_t, logit_scale)
    
    if visual_context is not None:
        loss_nce_context = loss_list['infoNCE'](visual_context, proj_context_text, None, None, logit_scale)
        loss_infoNCE = (loss_nce_color + loss_nce_type + loss_nce_motion + loss_nce_lang + loss_nce_context) / 5.0
    else:
        loss_infoNCE = (loss_nce_color + loss_nce_type + loss_nce_motion + loss_nce_lang) / 4.0

    pair_features = torch.cat([visual_embeds, proj_lang], dim=0)
    pair_labels = torch.cat([id_car, id_car], dim=0).long()
    loss_circle = loss_list['CircleLoss'](pair_features, pair_labels)
    loss_ce = F.cross_entropy(cls_logit, id_car.long())
    
    loss_total = 2.0 * loss_infoNCE + 0.2 * loss_circle + 0.5 * loss_ce

    return loss_total, loss_infoNCE, loss_circle, loss_ce, visual_embeds, proj_lang

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, loss_list, device, epoch, accumulation_steps=4):
    model.train()
    total_epoch_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=True)
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        with autocast():
            loss_total, loss_infoNCE, loss_circle, loss_ce, v_emb, t_emb = process_batch_loss(
                model, batch, device, loss_list, update_queue=True
            )
            loss_accum = loss_total / accumulation_steps
        
        scaler.scale(loss_accum).backward()
        
        #  [ĐÃ SỬA] Rút lệnh cập nhật Queue ra ngoài, để tất cả các micro-batch đều được lưu vào Queue
        model._dequeue_and_enqueue(v_emb.detach(), t_emb.detach())
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        real_loss_value = loss_total.item()
        total_epoch_loss += real_loss_value
        
        pbar.set_postfix({
            "Loss": f"{real_loss_value:.3f}",
            "NCE": f"{loss_infoNCE.item():.3f}",
            "Cir": f"{loss_circle.item():.3f}",
            "CE": f"{loss_ce.item():.3f}"
        })
        
    return total_epoch_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, loss_list, device, epoch):
    model.eval()
    total_val_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}", leave=True, colour='green')
    
    for batch in pbar:
        with autocast():
            loss_total, loss_infoNCE, loss_circle, loss_ce, _, _ = process_batch_loss(
                model, batch, device, loss_list, update_queue=False
            )
            
        real_loss_value = loss_total.item()
        total_val_loss += real_loss_value
        
        pbar.set_postfix({
            "Val_Loss": f"{real_loss_value:.3f}",
            "NCE": f"{loss_infoNCE.item():.3f}",
            "Cir": f"{loss_circle.item():.3f}",
            "CE": f"{loss_ce.item():.3f}"
        })
        
    return total_val_loss / len(dataloader)

# ==========================================
# 3. SETUP OPTIMIZER
# ==========================================
def build_optimizer_and_scheduler(model, total_steps, base_lr=1e-4):
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
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
    print(f" Bắt đầu quá trình huấn luyện trên {device.upper()}...")

    num_epochs = 300
    batch_size = 32
    accumulation_steps = 4  
    queue_size = 1024       
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

    print("Đang tải và phân chia Dataset...")
    full_dataset = CityFlowNLDataset(
        data_cfg=data_cfg, json_path="data/json/train_clean.json",
        text_emb_path="data/data/clip_text_tokens_extracted.pt",
        transform=train_transform, Random=True, type="train"
    )
    
    dataset_size = len(full_dataset)
    train_size = int(0.95 * dataset_size)
    val_size = dataset_size - train_size
    
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f" Đã chia Dataset: {train_size} Train | {val_size} Validation")

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, 
        shuffle=False, num_workers=4, pin_memory=True
    )

    print(f" Khởi tạo LiMaVLM với Queue Size: {queue_size}...")
    model = LiMaVLM(
        d_model=256, d_text_in=512, num_blocks=3, 
        num_classes=dataset_size, queue_size=queue_size
    ).to(device)

    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    
    optimizer, scheduler = build_optimizer_and_scheduler(model, total_steps, base_lr=1e-4)
    scaler = GradScaler()
    
    loss_list = {
        'infoNCE': XBMInfoNCELoss().to(device),
        'CircleLoss': CircleLoss(m=0.25, gamma=80).to(device)
    }

    print("\n Bắt đầu Training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, 
            loss_list, device, epoch, accumulation_steps
        )
        
        val_loss = validate(model, val_loader, loss_list, device, epoch)
        
        print(f"Kết thúc Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, "3blockslimavlm_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Đã cập nhật Best Model (Val Loss: {best_val_loss:.4f}) tại {best_path}")
            

        checkpoint_path = os.path.join(save_dir, f"3blockslimavlm_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f" Đã lưu Checkpoint định kỳ tại {checkpoint_path}\n")

if __name__ == "__main__":
    main()