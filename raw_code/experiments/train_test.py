import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import deform_conv2d
import torchvision.transforms as T
from PIL import Image
from einops import rearrange
from tqdm import tqdm
import cv2


# Yêu cầu cài đặt: pip install mamba-ssm einops
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
    print("⚠️ Cảnh báo: Không tìm thấy mamba_ssm. SelectiveScan sẽ không hoạt động trên GPU.")

# ==========================================
# 1. BACKBONE: DEFORMABLE + CAUSAL 3D CONV
# ==========================================
class DeformablePatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=16):
        super().__init__()
        self.stride = patch_size
        self.kernel_size = patch_size + 3
        self.padding = self.kernel_size // 2
        self.pre_norm = nn.GroupNorm(1, in_channels)
        
        self.offset_net = nn.Conv2d(in_channels, 2 * self.kernel_size**2, 
                                   kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        nn.init.constant_(self.offset_net.weight, 0.)
        nn.init.constant_(self.offset_net.bias, 0.)

        self.weight = nn.Parameter(torch.Tensor(embed_dim, in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        x = self.pre_norm(x)
        offsets = torch.tanh(self.offset_net(x)) * 2.0
        return deform_conv2d(x, offsets, self.weight, self.bias, stride=self.stride, padding=self.padding)

class CustomVideoBackbone(nn.Module):
    def __init__(self, d_model=256, num_frames=8, img_size=384, patch_size=16):
        super().__init__()
        self.patch_embed = DeformablePatchEmbedding(3, d_model, patch_size)
        self.dw_conv3d = nn.Conv3d(d_model, d_model, kernel_size=(3, 3, 3), padding=(0, 1, 1), groups=d_model)
        
        self.temp_embed = nn.Parameter(torch.zeros(1, d_model, num_frames))
        self.spatial_embed = nn.Parameter(torch.zeros(1, d_model, img_size//patch_size, img_size//patch_size))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_2d = x.transpose(1, 2).reshape(B * T, C, H, W)
        features_2d = self.patch_embed(x_2d)
        _, E, Hp, Wp = features_2d.shape
        
        patches_3d = features_2d.view(B, T, E, Hp, Wp).transpose(1, 2)
        patches_pad = F.pad(patches_3d, (0, 0, 0, 0, 2, 0)) 
        features_3d = F.gelu(self.dw_conv3d(patches_pad)) + patches_3d

        temp_pe = F.interpolate(self.temp_embed, size=T, mode='linear').transpose(1, 2).view(1, T, 1, 1, E)
        spatial_pe = F.interpolate(self.spatial_embed, size=(Hp, Wp), mode='bilinear').permute(0, 2, 3, 1).view(1, 1, Hp, Wp, E)
        
        out = features_3d.permute(0, 2, 3, 4, 1) + temp_pe + spatial_pe
        return self.norm(out) # [B, T, Hp, Wp, E]

# ==========================================
# 2. CORE: BI-DIRECTIONAL HIERARCHICAL MAMBA
# ==========================================
class MambaCore(nn.Module):
    """Lõi Mamba thuần túy tái sử dụng được"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        self.x_proj = nn.Linear(d_model, d_model + 2 * d_state)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(d_model, 1)))
        self.D_param = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_br, z_br = xz.chunk(2, dim=-1)
        x_conv = F.silu(self.conv1d(x_br.transpose(1, 2))[:, :, :L]).transpose(1, 2)

        proj = self.x_proj(x_conv)
        delta, B_mat, C_mat = torch.split(proj, [D, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())

        y = selective_scan_fn(
            x_conv.transpose(1, 2).contiguous(), delta.transpose(1, 2).contiguous(), 
            A, B_mat.transpose(1, 2).contiguous(), 
            C_mat.transpose(1, 2).contiguous(), self.D_param.float(),
            delta_softplus=True
        ).transpose(1, 2)

        return self.out_proj(y * F.silu(z_br))

class UltimateVLMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        # 1. Nhánh Không gian (MaTVLM)
        self.norm_spatial = nn.LayerNorm(d_model)
        self.spatial_mamba = MambaCore(d_model, d_state)

        # 2. Nhánh Thời gian Đa chiều (VL-Mamba + Cobra)
        self.norm_temporal = nn.LayerNorm(d_model)
        self.temporal_forward = MambaCore(d_model, d_state)
        self.temporal_backward = MambaCore(d_model, d_state)

    def forward(self, x, text_tokens):
        B, T, H, W, D = x.shape
        N = text_tokens.shape[1]

        # --- 1. SPATIAL SCAN (Chỉ quét HxW) ---
        x_spatial = x.contiguous().view(B * T, H * W, D)
        x_spatial = x_spatial + self.spatial_mamba(self.norm_spatial(x_spatial))
        x = x_spatial.view(B, T, H, W, D)

        # --- 2. TEMPORAL SCAN VỚI EARLY FUSION ---
        # Chuyển T ra sau cùng: Mỗi pixel trở thành một chuỗi thời gian độc lập
        x_temporal = x.permute(0, 2, 3, 1, 4).contiguous().view(B * H * W, T, D) 
        
        # Nhúng Text làm Prefix cho mọi pixel
        text_prefix = text_tokens.unsqueeze(1).unsqueeze(2).expand(B, H, W, N, D).contiguous().view(B * H * W, N, D)

        # Hướng Tiến (Forward: Frame 1 -> 8)
        seq_forward = torch.cat([text_prefix, x_temporal], dim=1) 
        out_forward = self.temporal_forward(self.norm_temporal(seq_forward))
        out_forward = out_forward[:, N:, :] # Bỏ Text, chỉ lấy lại phần Video

        # Hướng Lùi (Backward: Frame 8 -> 1)
        x_temporal_rev = torch.flip(x_temporal, dims=[1])
        seq_backward = torch.cat([text_prefix, x_temporal_rev], dim=1)
        out_backward = self.temporal_backward(self.norm_temporal(seq_backward))
        out_backward = torch.flip(out_backward[:, N:, :], dims=[1]) # Lật xuôi Video lại

        # Trộn đặc trưng Không gian + Tiến + Lùi
        x_temporal_out = x_temporal + out_forward + out_backward

        # Trả về không gian gốc [B, T, H, W, D]
        x_out = x_temporal_out.view(B, H, W, T, D).permute(0, 3, 1, 2, 4)
        return x_out

class LiMaVLM(nn.Module):
    def __init__(self, d_model=256, d_text=512, num_blocks=3):
        super().__init__()
        
        # Ánh xạ Text dimension xuống Model dimension
        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.blocks = nn.ModuleList([UltimateVLMambaBlock(d_model) for _ in range(num_blocks)])
        
        self.video_to_clip = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_text)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.hm_head = nn.Sequential(nn.Conv2d(d_model, d_model//2, 3, padding=1), nn.ReLU(), nn.Conv2d(d_model//2, 1, 1), nn.Sigmoid())
        self.sz_head = nn.Sequential(nn.Conv2d(d_model, d_model//2, 3, padding=1), nn.ReLU(), nn.Conv2d(d_model//2, 2, 1))
        self.off_head = nn.Sequential(nn.Conv2d(d_model, d_model//2, 3, padding=1), nn.ReLU(), nn.Conv2d(d_model//2, 2, 1))

    def forward(self, x, text_tokens):
        B, T, Hp, Wp, D = x.shape
        
        # Đưa Text về cùng chiều không gian với Mamba
        text_emb = self.text_proj(text_tokens)
        
        for block in self.blocks:
            x = block(x, text_emb)
            
        x_2d = rearrange(x, 'b t h w d -> (b t) d h w', t=T, h=Hp, w=Wp)
        return self.hm_head(x_2d), self.sz_head(x_2d), self.off_head(x_2d)

# ==========================================
# 3. DATASET: CITYFLOW-NL
# ==========================================
class CityFlowNLDataset(Dataset):
    def __init__(self, data_root, json_path, num_frames=8, img_size=384, down_ratio=8):
        self.data_root, self.num_frames, self.img_size, self.down_ratio = data_root, num_frames, img_size, down_ratio
        with open(json_path, 'r') as f:
            self.raw_data = json.load(f)
        self.samples = []
        for tid, info in self.raw_data.items():
            for text in info["nl"]:
                self.samples.append({"tid": tid, "frames": info["frames"], "boxes": info["boxes"], "text": text})
        
        self.text_embs = torch.load(os.path.join(data_root, "clip_text_tokens.pt"), map_location='cpu')
        self.transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self): return len(self.samples)

    def _draw_gaussian(self, heatmap, center, radius):
        diameter = 2 * radius + 1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        h = np.exp(-(x*x + y*y) / (2 * (diameter/6)**2))
        h_h, h_w = heatmap.shape
        ct_x, ct_y = int(center[0]), int(center[1])
        left, right = min(ct_x, radius), min(h_w - ct_x, radius + 1)
        top, bottom = min(ct_y, radius), min(h_h - ct_y, radius + 1)
        if left + right > 0 and top + bottom > 0:
            np.maximum(heatmap[ct_y-top:ct_y+bottom, ct_x-left:ct_x+right], h[radius-top:radius+bottom, radius-left:radius+right], out=heatmap[ct_y-top:ct_y+bottom, ct_x-left:ct_x+right])
        return heatmap

    def __getitem__(self, idx):
        s = self.samples[idx]
        text_emb = self.text_embs.get(s["text"].strip().lower(), torch.randn(32, 512)*1e-5)
        
        indices = np.linspace(0, len(s["frames"])-1, self.num_frames, dtype=int)
        hms = self.img_size // self.down_ratio 
        
        video, hm, sz, off = [], [], [], []
        for i in indices:
            # Thay bằng CV2:
            img_path = os.path.join(self.data_root, s["frames"][i])
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # OpenCV đọc BGR, phải chuyển sang RGB
            w_orig, h_orig = img_rgb.shape[1], img_rgb.shape[0]
            img_pil = Image.fromarray(img_rgb)
            video.append(self.transform(img_pil))
            
            h_target = np.zeros((hms, hms), dtype=np.float32)
            s_target = np.zeros((2, hms, hms), dtype=np.float32)
            o_target = np.zeros((2, hms, hms), dtype=np.float32)
            
            box = s["boxes"][i] 
            cx, cy = (box[0] + box[2]/2)/w_orig * hms, (box[1] + box[3]/2)/h_orig * hms
            bw, bh = box[2]/w_orig * hms, box[3]/h_orig * hms
            
            radius = max(1, int(math.sqrt(bw*bh)*0.15))
            self._draw_gaussian(h_target, (cx, cy), radius)
            
            xi, yi = int(cx), int(cy)
            if 0 <= xi < hms and 0 <= yi < hms:
                s_target[:, yi, xi] = np.log(np.array([bw, bh]) + 1e-6)
                o_target[:, yi, xi] = np.array([cx - xi, cy - yi])
            
            hm.append(torch.from_numpy(h_target).unsqueeze(0))
            sz.append(torch.from_numpy(s_target))
            off.append(torch.from_numpy(o_target))

        return {
            "video": torch.stack(video, dim=1), 
            "text_tokens": text_emb,
            "hm": torch.stack(hm), 
            "sz": torch.stack(sz), 
            "off": torch.stack(off)
        }

# ==========================================
# 4. LOSS FUNCTIONS & TRAIN LOOP
# ==========================================
def focal_loss(pred, target):
    pred = torch.clamp(pred, 1e-6, 1-1e-6)
    pos_loss = -(target == 1).float() * (1-pred).pow(2) * pred.log()
    neg_loss = -(target < 1).float() * (1-target).pow(4) * pred.pow(2) * (1-pred).log()
    return (pos_loss.sum() + neg_loss.sum()) / (target.eq(1).float().sum() + 1e-4)

def masked_l1_loss(pred, target, hm):
    mask = hm.eq(1).expand_as(pred)
    if mask.sum() == 0: return torch.tensor(0.).to(pred.device)
    return F.smooth_l1_loss(pred[mask], target[mask], reduction='sum') / (mask.sum() + 1e-4)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size, down_ratio = 384, 8
    hm_size = img_size // down_ratio 
    
    backbone = CustomVideoBackbone(d_model=256, num_frames=8, img_size=img_size, patch_size=8).to(device)
    vlm_head = LiMaVLM(d_model=256, d_text=512, num_blocks=2).to(device) 
    
    # KÍCH HOẠT TORCH COMPILE (Nếu bạn dùng PyTorch 2.0+)
    # Bỏ comment 2 dòng dưới nếu máy bạn hỗ trợ
    # backbone = torch.compile(backbone)
    # vlm_head = torch.compile(vlm_head)

    optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(vlm_head.parameters()), lr=1e-4)
    
    # 🚀 VŨ KHÍ 1: Tăng lại Batch Size lên 4 (Tận dụng GPU)
    ds = CityFlowNLDataset("./data/data", "./data/data/train-tracks.json", img_size=img_size, down_ratio=down_ratio)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "limavlm_best.pth")
    
    start_epoch = 0
    best_loss = float('inf')
    TOTAL_EPOCHS = 30
    
    
    if os.path.exists(best_model_path):
        print(f"🔄 Tìm thấy checkpoint tại {best_model_path}. Đang tải...")
        checkpoint = torch.load(best_model_path, map_location=device)
        backbone.load_state_dict(checkpoint['backbone_state'])
        vlm_head.load_state_dict(checkpoint['vlm_state'], strict=False) 
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f"✅ Đã tải thành công! Tiếp tục train từ Epoch {start_epoch + 1}")
    else:
        print("Bắt đầu train từ đầu (Từ Epoch 1)...")

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        backbone.train(); vlm_head.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
        epoch_loss = 0.0
        
        for batch in pbar:
            v = batch["video"].to(device)
            t_tokens = batch["text_tokens"].to(device) 
            
            t_hm = batch["hm"].to(device).view(-1, 1, hm_size, hm_size)
            t_sz = batch["sz"].to(device).view(-1, 2, hm_size, hm_size)
            t_off = batch["off"].to(device).view(-1, 2, hm_size, hm_size)
            
            optimizer.zero_grad()
            
            # 🚀 VŨ KHÍ 3: Mở không gian Autocast (Ép kiểu Float16 tự động)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                feats = backbone(v)
                p_hm, p_sz, p_off = vlm_head(feats, t_tokens) 
                
                # --- [SỬA Ở ĐÂY] ÉP KIỂU VỀ FLOAT32 TRƯỚC KHI TÍNH LOSS ---
                p_hm = p_hm.float()
                p_sz = p_sz.float()
                p_off = p_off.float()
                
                loss_hm = focal_loss(p_hm, t_hm)
                loss_sz = masked_l1_loss(p_sz, t_sz, t_hm)
                loss_off = masked_l1_loss(p_off, t_off, t_hm)

                B, T, H, W, D = feats.shape
                
                # Cần ép t_hm_mask về cùng kiểu float16 với feats để nhân không bị lỗi
                t_hm_mask = t_hm.view(B, T, H, W, 1).half() 
                v_target_temporal = (feats * t_hm_mask).sum(dim=(2, 3)) / (t_hm_mask.sum(dim=(2, 3)) + 1e-6)
                
                v_proj = vlm_head.video_to_clip(v_target_temporal)
                v_proj = F.normalize(v_proj, p=2, dim=-1).float() # Ép về Float32
                
                t_tokens_pos = batch["text_tokens"].to(device) 
                all_texts = list(ds.text_embs.values())
                random_indices = torch.randint(0, len(all_texts), (60,))
                t_tokens_neg = torch.stack([all_texts[i] for i in random_indices]).to(device)
                
                t_tokens_large = torch.cat([t_tokens_pos, t_tokens_neg], dim=0) 
                t_proj = F.normalize(t_tokens_large, p=2, dim=-1).float() # Ép về Float32
                
                # Các phép tính ma trận Loss sau đó sẽ an toàn tuyệt đối ở Float32
                sim = torch.einsum('vtd,bnd->vbtn', v_proj, t_proj) 
                score_v2t = sim.max(dim=3)[0].mean(dim=2) 
                
                logit_scale = vlm_head.logit_scale.float().exp()
                logits = logit_scale * score_v2t 
                
                labels = torch.arange(B, device=device)
                loss_contrastive = F.cross_entropy(logits, labels, label_smoothing=0.1)
                
                # Tổng hợp Loss
                total_loss = loss_hm + 0.1 * loss_sz + loss_off + 2.0 * loss_contrastive
            
            # Tính đạo hàm và cập nhật bình thường (Không dùng scaler)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm_head.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}", 
                "hm": f"{loss_hm.item():.3f}",
                "loss_c": f"{loss_contrastive.item():.4f}"
            })
            
        avg_epoch_loss = epoch_loss / len(dl)
            
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            # Đã tối giản code lưu checkpoint cho gọn
            torch.save({
                'epoch': epoch + 1,
                'backbone_state': backbone.state_dict(),
                'vlm_state': vlm_head.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_loss': best_loss,
            }, best_model_path)
            
            print(f"🌟 Epoch {epoch+1}: Kỷ lục Loss mới ({best_loss:.4f})! Đã lưu model.")
        else:
            print(f"ℹ️ Epoch {epoch+1}: Loss ({avg_epoch_loss:.4f}) không cải thiện (Best: {best_loss:.4f}).")
            
if __name__ == "__main__":
    train()