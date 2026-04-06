import torch
import torch.nn as nn
import torch.nn.functional as F

# BẮT BUỘC: Import hàm Scan tối ưu bằng CUDA của Mamba
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    raise ImportError("Vui lòng cài đặt mamba-ssm: pip install mamba-ssm")

class DualModulatedSTMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_text=512):
        """
        Lõi ST-Mamba tùy chỉnh cho mạng Li-Ma (Tối ưu hóa bằng CUDA).
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # 1. Bơm Ngữ nghĩa Cục bộ (Local Semantic Booster)
        self.conv1d = nn.Conv1d(
            in_channels=d_model, out_channels=d_model,
            kernel_size=3, padding=1, groups=d_model
        )

        # 2. Các lớp Tuyến tính dự phóng tham số SSM
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model)

        # 3. Khởi tạo Ma trận Hệ thống A 
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A)) 

        # Skip connection tĩnh
        self.D = nn.Parameter(torch.ones(d_model))

        # 4. Neuromodulation Gate (Cổng chất dẫn truyền Text)
        self.text_proj_B = nn.Linear(d_text, d_state)
        self.text_proj_C = nn.Linear(d_text, d_state)

    def forward(self, x, text_emb, view_scores):
        """
        Luồng Suy luận (Forward Pass) tích hợp CUDA.
        """
        batch, seq_len, _ = x.shape

        # --- Giai đoạn 1: Tiền xử lý Cục bộ ---
        x_conv = x.transpose(1, 2)
        x_conv = F.silu(self.conv1d(x_conv))
        x_conv = x_conv.transpose(1, 2) # [B, L, d_model]

        # --- Giai đoạn 2: Dự phóng Tham số SSM ---
        proj_out = self.x_proj(x_conv)
        delta, B_base, C_base = torch.split(
            proj_out, [self.d_model, self.d_state, self.d_state], dim=-1
        )

        # --- Giai đoạn 3: NEUROMODULATION GATE (Tiêm Text) ---
        G_B = torch.sigmoid(self.text_proj_B(text_emb)).unsqueeze(1) # [B, 1, d_state]
        G_C = torch.sigmoid(self.text_proj_C(text_emb)).unsqueeze(1)
        
        B = B_base * G_B # [B, L, d_state]
        C = C_base * G_C # [B, L, d_state]

        # --- Giai đoạn 4: VIEWPOINT-AWARE GATING ---
        # Chúng ta chạy softplus thủ công ở đây để có thể nhân với view_scores
        delta = F.softplus(delta) 
        delta = delta * view_scores # [B, L, d_model]

        # --- Giai đoạn 5: QUÉT KHÔNG - THỜI GIAN BẰNG CUDA ---
        # Để đưa vào selective_scan_fn, các tensor phải có shape [Batch, Dim, SeqLen]
        # và phải đảm bảo tính liên tục trong bộ nhớ (contiguous)
        
        u_cuda = x_conv.transpose(1, 2).contiguous()    # [B, d_model, L]
        delta_cuda = delta.transpose(1, 2).contiguous() # [B, d_model, L]
        B_cuda = B.transpose(1, 2).contiguous()         # [B, d_state, L]
        C_cuda = C.transpose(1, 2).contiguous()         # [B, d_state, L]
        
        A_cuda = -torch.exp(self.A_log.float()).contiguous() # [d_model, d_state]
        D_cuda = self.D.float().contiguous()                 # [d_model]

        # Gọi hàm Selective Scan từ thư viện Mamba.
        # QUAN TRỌNG: delta_softplus=False vì chúng ta đã tự tính Softplus ở Giai đoạn 4.
        # --- Giai đoạn 5: QUÉT KHÔNG - THỜI GIAN BẰNG CUDA ---
        
        # Xác định kiểu dữ liệu mục tiêu (thường là float16 nếu dùng AMP)
        target_dtype = x_conv.dtype 

        # Đưa về dạng [B, Dim, L] và đảm bảo liên tục + đúng kiểu dữ liệu
        u_cuda     = x_conv.transpose(1, 2).to(target_dtype).contiguous()
        delta_cuda = delta.transpose(1, 2).to(target_dtype).contiguous()
        B_cuda     = B.transpose(1, 2).to(target_dtype).contiguous()
        C_cuda     = C.transpose(1, 2).to(target_dtype).contiguous()
        
        # A và D nên được giữ ở float32 để tránh tràn số (Overflow)
        A_cuda = -torch.exp(self.A_log.float()).contiguous() # [d_model, d_state]
        D_cuda = self.D.float().contiguous()                 # [d_model]

        y_cuda = selective_scan_fn(
            u_cuda, 
            delta_cuda, 
            A_cuda, 
            B_cuda, 
            C_cuda, 
            D_cuda, 
            z=None, 
            delta_bias=None, 
            delta_softplus=False 
        )

        # y_cuda có dạng [B, d_model, L], chuyển lại thành [B, L, d_model]
        y = y_cuda.transpose(1, 2)

        # Không cần cộng thêm x_conv * self.D nữa vì hàm selective_scan_fn 
        # đã tự động làm việc đó ở bên trong kernel C++ rồi!
        
        return y_cuda.transpose(1, 2), G_B.mean(dim=(1, 2))