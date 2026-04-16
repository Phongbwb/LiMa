import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryAndPostProcessing(nn.Module):
    def __init__(self, embed_dim, d_text=512):
        super().__init__()
        self.embed_dim = embed_dim

        # =========================
        # (1) Learnable time constants (FIX 1: Differentiable Parameters)
        # =========================
        # Để raw values, sẽ dùng Sigmoid để chặn khoảng (bound) mượt mà ở forward
        self.tau_h_raw = nn.Parameter(torch.tensor(0.0)) # Khởi tạo 0 -> sigmoid(0) = 0.5
        self.tau_G_raw = nn.Parameter(torch.tensor(0.0))

        # =========================
        # (2) State update
        # =========================
        self.W_x = nn.Linear(embed_dim, embed_dim)
        self.W_G = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # [FIX 3]: Khởi tạo W_G cực nhỏ để tránh bùng nổ gradient do positive feedback
        nn.init.normal_(self.W_G.weight, mean=0.0, std=0.01)

        # confidence gate
        self.W_c = nn.Linear(embed_dim, 1)

        # dynamic tau (stronger)
        self.W_tau = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1)
        )

        # =========================
        # (3) Text conditioning (FiLM)
        # =========================
        self.text_proj = nn.Linear(d_text, embed_dim)
        self.gamma = nn.Linear(embed_dim, embed_dim)
        self.beta = nn.Linear(embed_dim, embed_dim)

        # =========================
        # (4) Normalization
        # =========================
        # [FIX 2]: Bỏ norm_h và norm_G vì h_t và G_t tự động được bounded bởi hàm tanh()
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, x_seq, text_emb, delta_t=1.0, states=None):
        """
        x_seq: [B, T, D]
        text_emb: [B, d_text]
        states: Tuple (h_prev, G_prev) - Truyền trạng thái giữa các chunk video
        """
        B, T, D = x_seq.shape
        device = x_seq.device

        # =========================
        # (0) PRECOMPUTE (VECTORIZATION)
        # =========================
        
        # Text conditioning
        text_feat = self.text_proj(text_emb)  # [B, D]
        gamma = torch.sigmoid(self.gamma(text_feat)).unsqueeze(1)  # [B, 1, D]
        beta = self.beta(text_feat).unsqueeze(1)                   # [B, 1, D]

        # Sequence projections
        x_proj_seq = self.W_x(x_seq)                  # [B, T, D]
        conf_seq = torch.sigmoid(self.W_c(x_seq))     # [B, T, 1]
        
        # [FIX 1]: Dùng Differentiable Bound (Sigmoid) thay vì Clamp cứng
        # Ép tau_h_base mượt mà vào khoảng [0.1, 20.1]
        tau_h_base = 0.1 + 20.0 * torch.sigmoid(self.tau_h_raw)
        
        dynamic_factor_seq = 1.0 + torch.sigmoid(self.W_tau(x_seq))  # [B, T, 1]
        tau_h_liquid_seq = tau_h_base / dynamic_factor_seq           # [B, T, 1]
        alpha_h_seq = torch.exp(-delta_t / tau_h_liquid_seq)         # [B, T, 1]

        # Ép tau_G mượt mà vào khoảng [1.0, 50.0]
        tau_G = 1.0 + 49.0 * torch.sigmoid(self.tau_G_raw)
        alpha_G = torch.exp(-delta_t / tau_G)

        # =========================
        # (1) Init states (FIX 4: Stateful chunking)
        # =========================
        if states is not None:
            h_t, G_t = states
        else:
            h_t = torch.zeros(B, D, device=device)
            G_t = torch.zeros(B, D, device=device)

        stable_feats = []

        # =========================
        # (2) Temporal loop (Siêu tốc - Đã gỡ bỏ bottleneck)
        # =========================
        for t in range(T):
            alpha_h = alpha_h_seq[:, t, :]    # [B, 1]
            x_p = x_proj_seq[:, t, :]         # [B, D]
            conf = conf_seq[:, t, :]          # [B, 1]

            # ----- state update -----
            # Hàm tanh() đã giới hạn giá trị của h_t trong khoảng [-1, 1]
            h_t = alpha_h * h_t + (1 - alpha_h) * torch.tanh(x_p + self.W_G(G_t))
            
            # G_t là trung bình cộng có trọng số (EMA) của h_t (vốn nằm trong [-1, 1]).
            # Theo nguyên lý toán học, G_t cũng sẽ vĩnh viễn bị chặn trong khoảng [-1, 1].
            G_t = alpha_G * G_t + (1 - alpha_G) * h_t

            # ----- confidence gating -----
            stable_feat = conf * h_t + (1 - conf) * G_t
            stable_feats.append(stable_feat)

        stable_feats_tensor = torch.stack(stable_feats, dim=1)

        # =========================
        # (3) Post-Processing
        # =========================
        out = stable_feats_tensor * gamma + beta
        out = out + x_seq
        out = self.norm_out(out)

        # Trả về output và state cuối cùng để chain vào chunk video tiếp theo
        return out, (h_t, G_t)