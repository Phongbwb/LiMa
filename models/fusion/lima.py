import torch
import torch.nn as nn
import torch.nn.functional as F

# Giả định bạn đã import các module đã thiết kế trước đó:
from models.sensing.neuromorphicfronted import NeuromorphicFrontend
from models.spatial_extraction.dyspatialextr import DynamicSpatialExtraction
from models.backbone.stmamba import DualModulatedSTMambaBlock
from models.localization.microlocalization import MicroLocalization
from models.memory.astropool import MemoryAndPostProcessing

class LiMaVLM(nn.Module):
    def __init__(self, in_channels=3, d_model=128, d_text=512, num_frames=8, img_size=112):
        """
        Kiến trúc hoàn chỉnh Li-Ma VLM tối ưu cho RTX 4060.
        """
        super().__init__()
        self.d_model = d_model
        self.num_frames = num_frames
        
        # Tính toán kích thước Patch (Giả sử patch_size = 4 ở lớp Spatial Extraction)
        self.patch_size = 4
        self.h_prime = img_size // self.patch_size
        self.w_prime = img_size // self.patch_size

        # ==========================================
        # 1. TẦNG TIỀN XỬ LÝ & TRÍCH XUẤT (BACKBONE)
        # ==========================================
        self.frontend = NeuromorphicFrontend(in_channels=in_channels, h=img_size, w=img_size)
        self.spatial_extractor = DynamicSpatialExtraction(in_channels=in_channels, embed_dim=d_model, patch_size=self.patch_size)
        
        # Module nhỏ dự đoán Hệ số góc nhìn (Viewpoint Scores) từ đặc trưng không gian
        self.view_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # ==========================================
        # 2. LÕI XỬ LÝ KHÔNG - THỜI GIAN (CORE)
        # ==========================================
        # (Đã tích hợp thư viện mamba-ssm để chạy CUDA)
        self.st_mamba = DualModulatedSTMambaBlock(d_model=d_model, d_state=16, d_text=d_text)

        # ==========================================
        # 3. TẦNG SOI CHI TIẾT & LOGIC (NECK & HEAD)
        # ==========================================
        # Soi chi tiết hạt mịn (Foveal Glimpse)
        self.micro_loc = MicroLocalization(embed_dim=d_model, num_parts=5)
        
        # Ổn định quỹ đạo thời gian liên tục (LNN Astrocytic Pool)
        self.memory_pool = MemoryAndPostProcessing(embed_dim=d_model, d_text=d_text)

        # ==========================================
        # 4. CÁC ĐẦU RA DỰ ĐOÁN (PREDICTION HEADS)
        # ==========================================
        # Đầu ra 1: Bounding Box [x_center, y_center, width, height]
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid() # Tọa độ chuẩn hóa [0, 1]
        )
        
        # Đầu ra 2: Đặc trưng Phân loại (Dùng cho H-SupCon Loss)
        self.cls_head = nn.Linear(d_model * 2, d_model) # Kết hợp đặc trưng toàn cục và cục bộ

    def forward(self, video_input, text_emb):
        """
        Luồng xuôi toàn trình của hệ thống.
        Args:
            video_input: Tensor [B, C, T, H, W]
            text_emb: Tensor [B, d_text]
        Returns:
            bbox_pred: [B, 4] (Tọa độ xe)
            gate_activation: [B] (Mức độ mở cổng Text để tính Auxiliary Loss)
            cls_features: [B, d_model] (Vector đặc trưng để đưa vào hàm H-SupCon)
        """
        B, C, T, H, W = video_input.shape

        # --- BƯỚC 1: LỌC NHIỄU SINH HỌC ---
        # Lọc nền tĩnh, chống nhòe
        clean_video = self.frontend(video_input)

        # --- BƯỚC 2: TRÍCH XUẤT KHÔNG GIAN ---
        # tokens: [B, SeqLen, d_model], trong đó SeqLen = T * H_prime * W_prime
        tokens, decouple_mask = self.spatial_extractor(clean_video)
        
        # Dự đoán hệ số góc nhìn [B, SeqLen, 1]
        view_scores = self.view_predictor(tokens)

        # --- BƯỚC 3: LÕI ST-MAMBA (VỚI CỔNG ĐIỀU BIẾN) ---
        # mamba_out: [B, SeqLen, d_model]
        # (Giả định bạn đã chỉnh sửa DualModulatedSTMambaBlock để trả về thêm gate_act)
        mamba_out, gate_act = self.st_mamba(tokens, text_emb, view_scores)

        # --- BƯỚC 4: SOI CHI TIẾT HẠT MỊN (MICRO-LOCALIZATION) ---
        # Để dùng RoIAlign, ta phải định dạng lại mamba_out về dạng hình ảnh 2D
        # [B, T * H' * W', d_model] -> [B, T, H', W', d_model] -> [B*T, d_model, H', W']
        # Lấy kích thước thực tế trực tiếp từ mamba_out
        # mamba_out có dạng [B, L, D]
        B_actual, L_actual, D_actual = mamba_out.shape
        
        # Dùng -1 để PyTorch tự động tính toán số khung hình T
        # Dựa trên B, H', W' và D thực tế
        feature_map_2d = mamba_out.view(B_actual, -1, self.h_prime, self.w_prime, D_actual)
        feature_map_2d = feature_map_2d.permute(0, 1, 4, 2, 3).reshape(B * T, self.d_model, self.h_prime, self.w_prime)
        
        # global_tokens cho MicroLoc: lấy trung bình không gian của mỗi frame
        global_tokens_per_frame = feature_map_2d.mean(dim=[2, 3]) # [B*T, d_model]
        
        # fine_features: Đặc trưng sắc nét của lốc máy, bánh xe... [B*T, d_model]
        fine_features, _, _ = self.micro_loc(feature_map_2d, global_tokens_per_frame)
        fine_features = fine_features.view(B, T, self.d_model) # Gom lại theo Batch và Time

        # --- BƯỚC 5: ỔN ĐỊNH QUỸ ĐẠO (ASTROCYTIC POOL) ---
        # Rút gọn mamba_out thành chuỗi thời gian bằng cách Global Average Pooling trên không gian
        temporal_seq = mamba_out.view(B, T, self.h_prime * self.w_prime, self.d_model).mean(dim=2) # [B, T, d_model]
        
        # Bám vết và suy luận logic Text
        tracked_features = self.memory_pool(temporal_seq, text_emb) # [B, T, d_model]

        # --- BƯỚC 6: XUẤT KẾT QUẢ ---
        # 6a. Lấy đặc trưng ở khung hình cuối cùng để dự đoán BBox
        last_tracked_feat = tracked_features[:, -1, :] # [B, d_model]
        bbox_pred = self.bbox_head(last_tracked_feat)
        
        # 6b. Tạo Vector Phân loại (Kết hợp Tổng thể + Chi tiết)
        # Nối đặc trưng quỹ đạo tổng thể và đặc trưng hạt mịn (hãng xe)
        last_fine_feat = fine_features[:, -1, :]
        fused_feat = torch.cat([last_tracked_feat, last_fine_feat], dim=-1) # [B, d_model * 2]
        cls_features = self.cls_head(fused_feat)

        return bbox_pred, gate_act, cls_features