import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import argparse
from tqdm import tqdm

# Import model và dataset của bạn
from models.lima import LiMaVLM
from experiments.utils.dataset import CityFlowNLDataset

# ==========================================
# 1. CÁC HÀM TÍNH LOSS
# ==========================================
def focal_loss_centernet(pred_hm, gt_hm, alpha=1.5, beta=4):
    """Focal Loss cho Heatmap Coarse (CenterNet)"""
    pred_hm = pred_hm.float()
    gt_hm = gt_hm.float()
    pred_hm = torch.clamp(pred_hm, 1e-4, 1 - 1e-4)
    pos = gt_hm.eq(1).float()
    neg = gt_hm.lt(1).float()
    neg_weight = torch.pow(1 - gt_hm, beta)
    pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, alpha) * pos
    neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, alpha) * neg_weight * neg
    num_pos = pos.sum()
    if num_pos == 0: return -neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos

def masked_l1_loss(pred, target, mask):
    """L1 Loss cho Size và Offset chỉ tại vị trí tâm"""
    mask = mask.unsqueeze(2) 
    loss = F.l1_loss(pred, target, reduction='none')
    loss = (loss * mask).sum() / (mask.sum() * 2 + 1e-3) 
    return loss

def hard_contrastive_and_ranking_loss(vid_feat, txt_pos, txt_neg, temperature=0.05, margin=0.2, label_smoothing=0.1):
    """InfoNCE Loss và Margin Ranking Loss có tích hợp Label Smoothing & Feature Dropout"""
    if vid_feat.requires_grad: 
        vid_feat = F.dropout(vid_feat, p=0.1, training=True)
        txt_pos = F.dropout(txt_pos, p=0.1, training=True)

    vid_feat = F.normalize(vid_feat, p=2, dim=-1)
    txt_pos = F.normalize(txt_pos, p=2, dim=-1)
    txt_neg = F.normalize(txt_neg, p=2, dim=-1)

    # InfoNCE Loss
    sim_v2t_inbatch = torch.matmul(vid_feat, txt_pos.t()) / temperature 
    sim_v2t_hard = torch.sum(vid_feat * txt_neg, dim=-1, keepdim=True) / temperature
    logits_v2t = torch.cat([sim_v2t_inbatch, sim_v2t_hard], dim=1) 
    labels = torch.arange(vid_feat.size(0), dtype=torch.long, device=vid_feat.device)
    
    loss_v2t = F.cross_entropy(logits_v2t, labels, label_smoothing=label_smoothing)
    logits_t2v = sim_v2t_inbatch.t()
    loss_t2v = F.cross_entropy(logits_t2v, labels, label_smoothing=label_smoothing)
    loss_cl = (loss_v2t + loss_t2v) / 2.0

    # Margin Ranking Loss
    sim_pos_score = torch.sum(vid_feat * txt_pos, dim=-1)
    sim_neg_score = torch.sum(vid_feat * txt_neg, dim=-1)
    target = torch.ones_like(sim_pos_score)
    loss_rank = F.margin_ranking_loss(sim_pos_score, sim_neg_score, target, margin=margin)

    return loss_cl, loss_rank

# ==========================================
# 2. CHẾ ĐỘ ĐÓNG BĂNG MẠNG VÀ OPTIMIZER
# ==========================================
def setup_training_stage(model, stage, lr):
    for p in model.parameters():
        p.requires_grad = False

    random_initialized_modules = [
        model.video_backbone.proj if hasattr(model.video_backbone, 'proj') else None,
        model.video_backbone.deform_patch_embed if hasattr(model.video_backbone, 'deform_patch_embed') else None,
        model.video_backbone.spatial_gate,
        model.video_backbone.temp_mlp,
        model.video_backbone.spatial_mlp,
        model.text_proj, 
        model.mamba,
        model.coarse_hm_head, 
        model.coarse_sz_head, 
        model.coarse_offset_head
    ]
    random_initialized_modules = [m for m in random_initialized_modules if m is not None]

    if stage == 1:
        print(" STAGE 1: Warm-up (Train các layer ngẫu nhiên | FREEZE Backbone & LNN)")
        for m in random_initialized_modules:
            for p in m.parameters(): p.requires_grad = True
        if hasattr(model.video_backbone, 'pe_scale'): model.video_backbone.pe_scale.requires_grad = True

    elif stage == 2:
        print(" STAGE 2: Joint Finetune (UNFREEZE Backbone Late Layers + Các layer ngẫu nhiên | FREEZE LNN)")
        for m in random_initialized_modules:
            for p in m.parameters(): p.requires_grad = True
        if hasattr(model.video_backbone, 'pe_scale'): model.video_backbone.pe_scale.requires_grad = True
        
        if hasattr(model.video_backbone, 'feature_extractor'):
            if len(model.video_backbone.feature_extractor) > 6:
                for p in model.video_backbone.feature_extractor[6].parameters(): p.requires_grad = True
            else:
                for p in model.video_backbone.feature_extractor[-1].parameters(): p.requires_grad = True

    elif stage == 3:
        print(" STAGE 3: (Bỏ qua Trajectory Comparator)")
        pass 
            
    elif stage == 4:
        print(" STAGE 4: Finetune ALL (End-to-End)")
        for p in model.parameters(): p.requires_grad = True
    else:
        raise ValueError(f"Stage {stage} không hợp lệ!")

    resnet_params = []
    if stage in [2, 4] and hasattr(model.video_backbone, 'feature_extractor'):
        resnet_params = [p for p in model.video_backbone.feature_extractor.parameters() if p.requires_grad]
        
    resnet_ids = set(id(p) for p in resnet_params)
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in resnet_ids]

    def split_decay_nodecay(params):
        decay = [p for p in params if p.ndim >= 2]
        nodecay = [p for p in params if p.ndim < 2]
        return decay, nodecay

    optim_groups = []
    if resnet_params: 
        res_decay, res_nodecay = split_decay_nodecay(resnet_params)
        optim_groups.append({'params': res_decay, 'lr': lr * 0.1, 'weight_decay': 1e-4})
        optim_groups.append({'params': res_nodecay, 'lr': lr * 0.1, 'weight_decay': 0.0})
        
    if other_params: 
        oth_decay, oth_nodecay = split_decay_nodecay(other_params)
        optim_groups.append({'params': oth_decay, 'lr': lr, 'weight_decay': 1e-4})
        optim_groups.append({'params': oth_nodecay, 'lr': lr, 'weight_decay': 0.0})

    return optim_groups

# ==========================================
# 3. EMA & CÁC HÀM TIỆN ÍCH BỔ TRỢ
# ==========================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow, self.backup = {}, {}
        for k, v in model.named_parameters():
            if v.requires_grad: self.shadow[k] = v.clone().detach()

    def update(self):
        for k, v in self.model.named_parameters():
            if v.requires_grad: self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply_shadow(self):
        self.backup = {}
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                self.backup[k] = v.clone()
                v.data.copy_(self.shadow[k])

    def restore(self):
        for k, v in self.model.named_parameters():
            if v.requires_grad: v.data.copy_(self.backup[k])

@torch.no_grad()
def custom_update_bn(loader, model, device='cuda'):
    print(" Đang chạy Custom Update BatchNorm cho SWA...")
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta: return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        if hasattr(module, 'num_batches_tracked'):
            module.num_batches_tracked *= 0

    for batch in loader:
        video = batch['video'].to(device, non_blocking=True)
        color_emb = batch['color_embedding'].to(device, non_blocking=True)
        type_emb = batch['type_embedding'].to(device, non_blocking=True)
        motion_emb = batch['motion_embedding'].to(device, non_blocking=True)
        context_emb = batch['context_embedding'].to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            model(video, color_emb, type_emb, motion_emb, context_emb)

    for module in momenta.keys():
        module.momentum = momenta[module]
    model.train(was_training)

@torch.no_grad()
def validate_limavlm(model, val_loader, device='cuda'):
    model.eval()
    print(" Đang chạy Validation...")
    
    total_samples = 0
    dummy_metric = 0.0 # TODO: Viết hàm IoU hoặc Accuracy của bạn ở đây
    
    for batch in val_loader:
        video = batch['video'].to(device, non_blocking=True)
        B = video.size(0)
        color_emb = batch['color_embedding'].to(device, non_blocking=True)
        type_emb = batch['type_embedding'].to(device, non_blocking=True)
        motion_emb = batch['motion_embedding'].to(device, non_blocking=True)
        context_emb = batch['context_embedding'].to(device, non_blocking=True)
        
        with autocast(dtype=torch.bfloat16):
            out = model(video, color_emb, type_emb, motion_emb, context_emb)
            
            # Tích hợp logic tìm Bbox tốt nhất tại bước Evaluation
            pred_bboxes = out.get("final_bboxes") # [B, T, 5, 4]
            pred_scores = out.get("frame_scores") # [B, T, 5]

            if pred_bboxes is not None and pred_scores is not None:
                B_out, T_out, _, _ = pred_bboxes.shape
                best_idx = torch.argmax(pred_scores, dim=2) # [B, T]
                
                b_idx = torch.arange(B_out, device=device).view(B_out, 1).expand(B_out, T_out)
                t_idx = torch.arange(T_out, device=device).view(1, T_out).expand(B_out, T_out)
                
                best_bboxes = pred_bboxes[b_idx, t_idx, best_idx] # [B, T, 4]
                best_scores = pred_scores[b_idx, t_idx, best_idx] # [B, T]
                
                # TODO: So sánh best_bboxes với Ground Truth (batch['gt_boxes']) bằng hàm IoU
                pass
                
        total_samples += B

    print(f" Kết quả Validation: Dummy Metric = {dummy_metric:.4f}")
    return dummy_metric

# ==========================================
# 4. TRAIN LOOP CHÍNH
# ==========================================
def train(model, train_loader, val_loader, args, device='cuda'):
    model.to(device)

    optim_groups = setup_training_stage(model, args.stage, args.lr)
    optimizer = optim.AdamW(optim_groups) 

    steps_per_epoch = len(train_loader)
    warmup_epochs = max(1, args.epochs // 10) if args.start_epoch == 1 else 0
    warmup_steps = warmup_epochs * steps_per_epoch
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    t_max_steps = max(1, (args.epochs - warmup_epochs) * steps_per_epoch)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=t_max_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    scaler = GradScaler()
    ema = EMA(model)
    best_metric = 0.0
    epochs_no_improve = 0 

    use_swa = (args.stage == 4)
    if use_swa:
        swa_model = AveragedModel(model)
        swa_start_epoch = int(args.epochs * 0.7) 
        swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.5)

    lambda_hm      = 1.0 if args.stage in [1, 2, 4] else 0.0
    lambda_sz      = 0.1 if args.stage in [1, 2, 4] else 0.0
    lambda_offset  = 1.0 if args.stage in [1, 2, 4] else 0.0
    lambda_cl      = 2.5 if args.stage in [1, 2, 4] else 0.0 
    lambda_offset_backbone = 0.01 if args.stage in [1, 2, 4] else 0.0 

    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        total_loss_epoch = 0

        pbar = tqdm(train_loader, desc=f"Stage {args.stage} | Epoch [{epoch}/{args.epochs}]", leave=False, dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad() 
            
            video = batch['video'].to(device, non_blocking=True)
            gt_hm = batch['hm'].to(device, non_blocking=True)       
            gt_sz = batch['sz'].to(device, non_blocking=True)       
            gt_offset = batch['offset'].to(device, non_blocking=True) 
            reg_mask = batch['reg_mask'].to(device, non_blocking=True) 

            color_emb = batch['color_embedding'].to(device, non_blocking=True)
            type_emb = batch['type_embedding'].to(device, non_blocking=True)
            motion_emb = batch['motion_embedding'].to(device, non_blocking=True)
            context_emb = batch['context_embedding'].to(device, non_blocking=True)
            
            hn_color_emb = batch['hn_color_embedding'].to(device, non_blocking=True)
            hn_type_emb = batch['hn_type_embedding'].to(device, non_blocking=True)

            with autocast(dtype=torch.bfloat16):
                out = model(video, color_emb, type_emb, motion_emb, context_emb)
                
                # --- TÍCH HỢP TÌM BEST BBOX ---
                pred_bboxes = out.get("final_bboxes") # [B, T, 5, 4]
                pred_scores = out.get("frame_scores") # [B, T, 5]

                if pred_bboxes is not None and pred_scores is not None:
                    B_out, T_out, _, _ = pred_bboxes.shape
                    best_idx = torch.argmax(pred_scores, dim=2) # [B, T]
                    
                    b_idx = torch.arange(B_out, device=device).view(B_out, 1).expand(B_out, T_out)
                    t_idx = torch.arange(T_out, device=device).view(1, T_out).expand(B_out, T_out)
                    
                    best_bboxes = pred_bboxes[b_idx, t_idx, best_idx] # [B, T, 4]
                    best_scores = pred_scores[b_idx, t_idx, best_idx] # [B, T]
                    
                    # (Tùy chọn) Nếu bạn muốn tính thêm GIoU Loss giữa best_bboxes và GT Bboxes:
                    # loss_bbox = compute_giou(best_bboxes, gt_bboxes) 
                # -------------------------------

                l_hm = focal_loss_centernet(out["hm_coarse"], gt_hm) if lambda_hm > 0 else torch.tensor(0.0, device=device)
                l_sz = masked_l1_loss(out["sz_coarse"], gt_sz, reg_mask) if lambda_sz > 0 else torch.tensor(0.0, device=device)
                l_offset = masked_l1_loss(out["offset_coarse"], gt_offset, reg_mask) if lambda_offset > 0 else torch.tensor(0.0, device=device)
                offset_loss = out["loss_off"].mean() if lambda_offset_backbone > 0 else torch.tensor(0.0, device=device)
                    
                mamba_out = out.get("mamba_out_grounded") 
                if lambda_cl > 0 and mamba_out is not None:
                    B_m, T_m, H, W, D = mamba_out.shape
                    loss_cl = 0.0; loss_cl_rank = 0.0; valid_items = 0
                    
                    noise_std = 0.01
                    c_emb_noisy = color_emb + torch.randn_like(color_emb) * noise_std
                    t_emb_noisy = type_emb + torch.randn_like(type_emb) * noise_std

                    txt_color_pos = model.text_proj(c_emb_noisy).mean(dim=1) 
                    txt_type_pos = model.text_proj(t_emb_noisy).mean(dim=1) 
                    txt_color_neg = model.text_proj(hn_color_emb).mean(dim=1) 
                    txt_type_neg = model.text_proj(hn_type_emb).mean(dim=1) 
                    
                    for b in range(B_m):
                        for t in range(T_m):
                            mask_bt = reg_mask[b, t]
                            if mask_bt.sum() == 0: continue 
                            
                            pos_y, pos_x = torch.where(mask_bt == 1)
                            car_feature = mamba_out[b, t, pos_y[0], pos_x[0], :].unsqueeze(0)
                            
                            l_cl_color, l_rank_color = hard_contrastive_and_ranking_loss(
                                car_feature, txt_color_pos[b].unsqueeze(0), txt_color_neg[b].unsqueeze(0)
                            )
                            l_cl_type, l_rank_type = hard_contrastive_and_ranking_loss(
                                car_feature, txt_type_pos[b].unsqueeze(0), txt_type_neg[b].unsqueeze(0)
                            )
                            
                            loss_cl += (l_cl_color + l_cl_type) / 2.0
                            loss_cl_rank += (l_rank_color + l_rank_type) / 2.0
                            valid_items += 1
                            
                    if valid_items > 0:
                        l_cl, l_cl_rank = loss_cl / valid_items, loss_cl_rank / valid_items
                    else:
                        l_cl = l_cl_rank = torch.tensor(0.0, device=device)
                else:
                    l_cl = l_cl_rank = torch.tensor(0.0, device=device)

                # Tổng hợp Loss
                loss = (lambda_hm * l_hm) + (lambda_sz * l_sz) + (lambda_offset * l_offset) + \
                       (lambda_cl * l_cl) + (lambda_cl * l_cl_rank) + (lambda_offset_backbone * offset_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            total_loss_epoch += loss.item()
            pbar.set_postfix({'L': f"{loss.item():.2f}", 'Hm': f"{l_hm.item():.2f}" if lambda_hm > 0 else "-", 'CL': f"{l_cl.item():.2f}" if lambda_cl > 0 else "-"})

        avg_loss = total_loss_epoch / len(train_loader)
        print(f" Epoch {epoch}/{args.epochs} | Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[-1]['lr']:.6f}")

        # VALIDATION VÀ EARLY STOPPING
        if epoch % args.val_interval == 0 or epoch == 1 or epoch == args.epochs:
            if args.stage >= 3:
                ema.apply_shadow()
                current_metric = validate_limavlm(model, val_loader, device)
                ema.restore()

                if current_metric >= best_metric: # >= để ít nhất save ở epoch 1
                    best_metric = current_metric
                    epochs_no_improve = 0 
                    torch.save(model.state_dict(), os.path.join(args.data_root, f"best_limavlm_stage{args.stage}.pth"))
                    print(f" LƯU BEST MODEL (Metric: {best_metric:.4f})")
                else:
                    epochs_no_improve += 1
                    print(f" Không tăng cường trên Validation. Patience: {epochs_no_improve}/{args.patience}")
                    
                if epochs_no_improve >= args.patience:
                    print(f" Kích hoạt Early Stopping tại Epoch {epoch}. Best Metric: {best_metric:.4f}")
                    break 
            else:
                torch.save(model.state_dict(), os.path.join(args.data_root, f"best_limavlm_stage{args.stage}.pth"))

        checkpoint_path = os.path.join(args.data_root, f"checkpoint_stage{args.stage}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    if use_swa:
        print(" Đang tổng hợp SWA Model...")
        custom_update_bn(train_loader, swa_model, device)
        torch.save(swa_model.state_dict(), os.path.join(args.data_root, f"best_limavlm_stage4_SWA.pth"))
        print(" Lưu thành công mô hình SWA!")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4], help="Giai đoạn huấn luyện (1-4)")
    parser.add_argument("--data_root", type=str, default="./data/data")
    parser.add_argument("--train_json", type=str, default="train-tracks.json")
    parser.add_argument("--val_json", type=str, default="test-tracks.json")
    parser.add_argument("--text_emb", type=str, default="clip_text_tokens_extracted_optimized.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--start_epoch", type=int, default=1) 
    parser.add_argument("--val_interval", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5, help="Số epoch tối đa kích hoạt Early Stopping")
    args = parser.parse_args()

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Thiết bị: {device} |  Batch Size: {args.batch_size}")
    
    TRAIN_JSON = os.path.join(args.data_root, args.train_json)
    VAL_JSON = os.path.join(args.data_root, args.val_json)
    TEXT_EMB_PATH = os.path.join(args.data_root, args.text_emb)
    
    train_dataset = CityFlowNLDataset(TRAIN_JSON, args.data_root, TEXT_EMB_PATH, max_frames=8, img_size=384)
    val_dataset = CityFlowNLDataset(VAL_JSON, args.data_root, TEXT_EMB_PATH, max_frames=8, img_size=384)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    model = LiMaVLM(d_model=256, d_text=512, num_blocks=4)

    if args.resume is not None and os.path.exists(args.resume):
        print(f" Đang tải checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        
    print(f"\n BẮT ĐẦU TRAINING - STAGE {args.stage}\n")
    train(model=model, train_loader=train_loader, val_loader=val_loader, args=args, device=device)