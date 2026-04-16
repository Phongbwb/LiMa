import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

class MicroLocalization(nn.Module):
    def __init__(self, embed_dim, num_parts=5):
        super().__init__()
        self.num_parts = num_parts
        self.embed_dim = embed_dim

        # =========================
        # (1) Predict parts: x, y, visibility, scale
        # =========================
        self.part_predictor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_parts * 4)
        )

        # =========================
        # (2) Static Sobel Filters (FIX 2: Buffer thay vì Parameter)
        # =========================
        sobel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]]
        )
        sobel_y = torch.tensor(
            [[-1., -2., -1.],
             [0.,  0.,  0.],
             [1.,  2.,  1.]]
        )

        # Đăng ký dưới dạng Buffer để Pytorch đưa lên GPU cùng Model 
        # nhưng KHÔNG update bởi Optimizer (tránh Filter Drift)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        # =========================
        # (3) Channel Attention
        # =========================
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 8, embed_dim, 1),
            nn.Sigmoid()
        )

        # =========================
        # (4) Part Attention & Positional Embedding (FIX 3)
        # =========================
        # Giúp MultiheadAttention phân biệt được 5 parts khác nhau
        self.part_pos_embed = nn.Parameter(torch.randn(1, num_parts, embed_dim) * 0.02)
        
        self.part_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True
        )

        # =========================
        # (5) Output projection & Norm
        # =========================
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, feature_map, global_tokens):
        """
        feature_map: [B*T, C, H, W]
        global_tokens: [B*T, C]
        """
        device = feature_map.device
        bt, c, h, w = feature_map.shape

        # =========================
        # (1) Predict parts
        # =========================
        part_params = self.part_predictor(global_tokens)
        part_params = part_params.view(bt, self.num_parts, 4)

        coords = torch.sigmoid(part_params[..., :2])        # [0,1]
        visibility = torch.sigmoid(part_params[..., 2:3])   # [0,1]
        scale = torch.sigmoid(part_params[..., 3:4])        # [0,1]

        # adaptive box size
        box_size = 0.1 + 0.2 * scale  # [0.1 -> 0.3]

        # =========================
        # (2) Vectorized ROI Align (FIX 1: Safe Clamping)
        # =========================
        batch_idx = torch.arange(bt, device=device).view(-1, 1).repeat(1, self.num_parts)

        cx = coords[..., 0] * w
        cy = coords[..., 1] * h

        half_w = box_size[..., 0] * w / 2
        half_h = box_size[..., 0] * h / 2

        # Kẹp (Clamp) giá trị NGAY TẠI ĐÂY để tránh lỗi Autograd inplace
        x1 = (cx - half_w).clamp(min=0, max=w - 1)
        y1 = (cy - half_h).clamp(min=0, max=h - 1)
        x2 = (cx + half_w).clamp(min=0, max=w - 1)
        y2 = (cy + half_h).clamp(min=0, max=h - 1)

        # Gom lại thành tensor chuẩn bị cho ROI Align
        rois = torch.stack([
            batch_idx.flatten(),
            x1.flatten(),
            y1.flatten(),
            x2.flatten(),
            y2.flatten()
        ], dim=1)

        part_features = roi_align(
            feature_map,
            rois,
            output_size=(7, 7),
            spatial_scale=1.0,
            aligned=True
        )

        flat_num_parts = bt * self.num_parts

        # =========================
        # (3) Edge-aware enhancement (Vectorized)
        # =========================
        sobel_x = self.sobel_x.repeat(c, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(c, 1, 1, 1)

        grad_x = F.conv2d(part_features, sobel_x, padding=1, groups=c)
        grad_y = F.conv2d(part_features, sobel_y, padding=1, groups=c)

        edge_map = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        edge_map = edge_map / (edge_map.mean(dim=[2, 3], keepdim=True) + 1e-6)

        attn = self.channel_attn(part_features)
        part_features = part_features * attn * (1 + edge_map)

        pooled_parts = F.adaptive_avg_pool2d(part_features, 1).view(bt, self.num_parts, c)
        pooled_parts = pooled_parts * visibility

        # =========================
        # (4) Part attention (FIX 3: Positional Embedding)
        # =========================
        # Cộng Positional Embedding để mạng biết phân biệt 5 vùng với nhau
        pooled_parts = pooled_parts + self.part_pos_embed
        
        parts, _ = self.part_attn(pooled_parts, pooled_parts, pooled_parts)
        out = parts.mean(dim=1)

        # =========================
        # (5) Residual + projection + Norm
        # =========================
        out = self.out_proj(out)
        out = self.norm_out(out + global_tokens) 

        return out, coords, visibility

    # =========================
    # HÀM PHỤ TRỢ DÙNG KHI TRAINING
    # =========================
    def get_diversity_loss(self, coords, margin=0.1):
        """
        Tính Loss để ép 5 parts phải tản ra xa nhau, chống mode-collapse.
        coords: [B*T, num_parts, 2] (giá trị đã normalize [0, 1])
        """
        bt, n, _ = coords.shape
        loss_div = 0.0
        
        # Lặp qua các cặp parts (vd: part 0 và part 1, part 0 và part 2...)
        for i in range(n):
            for j in range(i + 1, n):
                # Khoảng cách L2 (Euclidean distance)
                dist = torch.norm(coords[:, i, :] - coords[:, j, :], p=2, dim=-1)
                
                # Nếu khoảng cách < margin (đang đè lên nhau), ta phạt (penalize)
                # Dùng F.relu để nếu dist >= margin thì loss = 0
                loss_div += F.relu(margin - dist).mean()
                
        # Trung bình hóa cho số cặp
        num_pairs = n * (n - 1) / 2
        return loss_div / num_pairs