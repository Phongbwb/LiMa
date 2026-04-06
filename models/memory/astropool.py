import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryAndPostProcessing(nn.Module):
    def __init__(self, embed_dim, d_text=512):
        """
        Module hậu xử lý ổn định quỹ đạo và suy luận logic.
        Args:
            embed_dim (int): Số chiều đặc trưng từ ST-Mamba.
            d_text (int): Số chiều của đặc trưng văn bản.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Astrocytic Pool: Tham số thời gian khả vi (Learnable Time Constants) [cite: 29]
        # Khởi tạo log_tau để đảm bảo tau luôn dương
        self.log_tau_h = nn.Parameter(torch.log(torch.tensor(0.5))) # Nơ-ron phản ứng nhanh
        self.log_tau_G = nn.Parameter(torch.log(torch.tensor(10.0))) # Tế bào hình sao giữ bối cảnh

        # Các lớp chuyển đổi trạng thái [cite: 29]
        self.W_x = nn.Linear(embed_dim, embed_dim)
        self.W_G = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_c = nn.Linear(embed_dim, 1) # Cổng tự tin (Confidence Gate) [cite: 30]

        # 2. Cortical Re-entrant Feedback (Tia kỳ vọng) [cite: 30]
        self.expectation_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        # 3. Lightweight Cross-Attention (Xử lý logic văn bản phức tạp) 
        # Chỉ dùng 1 head để tiết kiệm VRAM 
        self.text_cross_attn = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.text_proj = nn.Linear(d_text, embed_dim)
        
        self.W_tau = nn.Linear(embed_dim, 1)

    def forward(self, x_seq, text_emb, delta_t=1.0):
            """
            x_seq: [B, T, EmbedDim] - Đặc trưng từ các bước trước
            text_emb: [B, d_text] - Đặc trưng văn bản
            """
            batch, seq_len, _ = x_seq.size()
            device = x_seq.device

            # Chuẩn bị đặc trưng văn bản cho Cross-Attention 
            text_feat = self.text_proj(text_emb).unsqueeze(1) # [B, 1, EmbedDim]

            # Khởi tạo trạng thái ẩn cho Astrocytic Pool
            h_t = torch.zeros(batch, self.embed_dim, device=device)
            G_t = torch.zeros(batch, self.embed_dim, device=device)
            
            # Tau_G (Astrocyte) là hằng số, có thể tính ngoài vòng lặp để tiết kiệm chi phí
            tau_G = torch.exp(self.log_tau_G)
            alpha_G = torch.exp(-delta_t / tau_G)

            final_outputs = []

            # Vòng lặp thời gian để ổn định quỹ đạo 
            for t in range(seq_len):
                x_t = x_seq[:, t, :]

                # --- SỬA LỖI Ở ĐÂY: Đưa phần tính toán LNN vào trong vòng lặp ---
                # dynamic_factor_h phải được tính cho từng x_t cụ thể ở thời điểm t
                # Đổi self.W_tau_h thành self.W_tau (đúng như tên đã khai báo trong __init__)
                dynamic_factor_h = 1.0 + torch.sigmoid(self.W_tau(x_t)) 
                tau_h_liquid = torch.exp(self.log_tau_h) / dynamic_factor_h
                alpha_h = torch.exp(-delta_t / tau_h_liquid)
                # ----------------------------------------------------------------

                # --- Astrocytic Pool Update ---
                # Trạng thái nơ-ron ht bám sát chuyển động nhanh
                h_t = alpha_h * h_t + (1 - alpha_h) * torch.tanh(self.W_x(x_t) + self.W_G(G_t))
                # Trạng thái tế bào hình sao Gt ghi nhớ bối cảnh ngã tư
                G_t = alpha_G * G_t + (1 - alpha_G) * h_t

                # --- Cortical Re-entrant Feedback (Tia kỳ vọng) ---
                # Tính độ tự tin để chống chớp tắt (flickering)
                conf = torch.sigmoid(self.W_c(x_t))
                # Nội suy: Nếu xe bị che khuất (conf thấp), ưu tiên dùng trí nhớ Gt 
                stable_feat = conf * h_t + (1 - conf) * G_t
                
                # --- Lightweight Cross-Attention ---
                # So khớp logic: "xe máy" + "chở thùng xốp"
                # stable_feat đóng vai trò Query, text_feat đóng vai trò Key/Value
                q = stable_feat.unsqueeze(1)
                attn_out, _ = self.text_cross_attn(q, text_feat, text_feat)
                
                final_outputs.append(attn_out)

            # Kết hợp lại thành chuỗi video hoàn chỉnh [B, T, EmbedDim]
            return torch.cat(final_outputs, dim=1)