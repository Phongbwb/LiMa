import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class DualModulatedSTMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_text=512):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # --- 1. Normalization ---
        self.norm = nn.LayerNorm(d_model)

        # --- 2. Input Projection (TẠO NHÁNH z) ---
        # [FIX 3]: Mở rộng đầu ra gấp đôi để chia làm 2 nhánh: x_branch (vào SSM) và z_branch (Skip connection gating)
        self.in_proj = nn.Linear(d_model, d_model * 2)

        # --- 3. Local Conv1D (CAUSAL CONVOLUTION) ---
        # [FIX 2]: Bỏ padding=1 để tránh nhìn trộm tương lai. Padding sẽ được tự code trong forward.
        self.conv1d = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=3, 
            padding=0, # <-- Sửa ở đây
            groups=d_model
        )

        # --- 4. SSM Projection ---
        self.x_proj = nn.Linear(d_model, d_model + 2*d_state)

        # --- 5. System matrix A (KEEP STATIC) ---
        A_init = torch.arange(1, d_state+1).float().repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A_init))

        # --- 6. System vector D ---
        self.D_param = nn.Parameter(torch.ones(d_model))

        # --- 7. TEXT MODULATION ---
        self.text_proj_B = nn.Linear(d_text, d_state)
        self.text_proj_C = nn.Linear(d_text, d_state)
        
        # [FIX 1]: Khởi tạo bias để tránh Vanishing Delta
        self.text_proj_delta = nn.Linear(d_text, d_model)
        nn.init.constant_(self.text_proj_delta.bias, 1.0) # Khởi tạo thiên lệch = 1.0 -> Sigmoid ~ 0.73

        # --- 8. VIEWPOINT MLP ---
        self.view_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        # [FIX 1]: Khởi tạo bias lớp cuối để tránh Vanishing Delta
        nn.init.constant_(self.view_mlp[-1].bias, 1.0) 

        # --- 9. OUTPUT PROJECTION ---
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, text_emb, view_scores):
        """
        x: [B, L, D]
        text_emb: [B, d_text]
        view_scores: [B, L, 1]
        """
        B_dim, L, D = x.shape
        residual = x

        # --- Normalization ---
        x_norm = self.norm(x)

        # --- [FIX 3]: Tách nhánh (x và z) ---
        xz = self.in_proj(x_norm)                 # [B, L, 2D]
        x_branch, z_branch = xz.chunk(2, dim=-1)  # Mỗi nhánh [B, L, D]

        # --- [FIX 2]: Causal Conv1D ---
        x_branch = x_branch.transpose(1, 2)       # [B, D, L]
        # Nhân quả (Causal padding): Thêm (kernel_size - 1) = 2 số 0 vào bên trái, 0 số không vào bên phải
        x_pad = F.pad(x_branch, (2, 0))
        x_conv = self.conv1d(x_pad)               # Cắt đi để giữ đúng độ dài L: [B, D, L]
        x_conv = x_conv[:, :, :L] 
        x_conv = F.silu(x_conv).transpose(1, 2)   # [B, L, D]

        # --- Project to SSM params ---
        proj = self.x_proj(x_conv)                # [B, L, D + 2*d_state]
        delta_base, B_base, C_base = torch.split(proj, [D, self.d_state, self.d_state], dim=-1)

        # --- Text Modulation ---
        G_B = torch.sigmoid(self.text_proj_B(text_emb)).unsqueeze(1)         
        G_C = torch.sigmoid(self.text_proj_C(text_emb)).unsqueeze(1)         
        
        # Nhờ bias = 1.0, G_delta ban đầu sẽ quanh mức 0.73 thay vì 0.5
        G_delta = torch.sigmoid(self.text_proj_delta(text_emb)).unsqueeze(1) 

        B_mod = B_base * G_B         
        C_mod = C_base * G_C         
        delta_mod = delta_base * G_delta 

        # --- View Modulation ---
        # Nhờ bias = 1.0, view_feat ban đầu sẽ quanh mức 0.73
        view_feat = torch.sigmoid(self.view_mlp(view_scores))  
        delta_mod = delta_mod * view_feat

        # --- A matrix & Inputs for Scan ---
        A_mat = -torch.exp(self.A_log.float()).contiguous()  
        
        u = x_conv.transpose(1, 2).contiguous().float()          
        delta_mod = delta_mod.transpose(1, 2).contiguous().float() 
        B_mod = B_mod.transpose(1, 2).contiguous().float()       
        C_mod = C_mod.transpose(1, 2).contiguous().float()       
        D_vec = self.D_param.float().contiguous()                

        # --- Run SSM scan ---
        y = selective_scan_fn(
            u, delta_mod, A_mat, B_mod, C_mod, D_vec,
            z=None, delta_bias=None, delta_softplus=True
        )
        y = y.transpose(1, 2)  # [B, L, D]

        # --- [FIX 3]: Gating với nhánh z (Standard Mamba Design) ---
        # Mamba tạo ra tính phi tuyến cực mạnh ở bước này
        y = y * F.silu(z_branch)

        # --- Output Projection & Residual ---
        y = self.out_proj(y)
        y = y + residual

        # Tính mean của cổng z để track xem mạng có bị bão hòa không (tùy chọn)
        z_gate_mean = torch.sigmoid(z_branch).mean(dim=(1, 2))
        
        return y, z_gate_mean