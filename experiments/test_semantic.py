import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.train_test import CustomVideoBackbone, LiMaVLM, CityFlowNLDataset

def test_semantic_alignment(checkpoint_path="./checkpoints/limavlm_best.pth", data_root="./data/data", json_path="./data/data/train-tracks.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Bắt đầu test ngữ nghĩa trên thiết bị: {device}")

    img_size, down_ratio = 384, 8
    hm_size = img_size // down_ratio 
    
    # 1. Khởi tạo mô hình
    backbone = CustomVideoBackbone(d_model=256, num_frames=8, img_size=img_size, patch_size=8).to(device)
    vlm_head = LiMaVLM(d_model=256, d_text=512, num_blocks=2).to(device)
    
    # 2. Tải trọng số đã train
    if not os.path.exists(checkpoint_path):
        print(f"❌ Không tìm thấy checkpoint tại {checkpoint_path}. Vui lòng train mô hình trước!")
        return
        
    print(f"🔄 Đang tải trọng số từ {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(checkpoint['backbone_state'])
    vlm_head.load_state_dict(checkpoint['vlm_state'])
    
    backbone.eval()
    vlm_head.eval()
    print("✅ Đã tải thành công và chuyển sang chế độ Eval.")

    # 3. Tải Dữ liệu thật (Dùng luôn tập train hoặc validation nếu bạn có file json riêng)
    ds = CityFlowNLDataset(data_root, json_path, img_size=img_size, down_ratio=down_ratio)
    
    # Lấy 1 batch = 4 để so sánh chéo (Video 1 vs Text 1, 2, 3, 4)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, drop_last=True)
    
    batch = next(iter(dl)) # Rút thử 1 Batch đầu tiên
    
    v = batch["video"].to(device)
    t_tokens = batch["text_tokens"].to(device) 
    
    # Lấy Ground Truth Heatmap làm Mask (Giống hệt cách làm lúc Train)
    # hm_shape của dataset là: [B, T, 1, Hp, Wp] -> Reshape về [B, T, Hp, Wp, 1]
    t_hm = batch["hm"].to(device)
    B, T, _, H, W = t_hm.shape
    t_hm_mask = t_hm.view(B, T, H, W, 1).half() # Dùng half/bfloat16 để nhân với feats

    with torch.no_grad():
        with torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', dtype=torch.float16):
            # A. Lấy đặc trưng toàn ảnh từ Backbone
            feats = backbone(v) # [B, T, Hp, Wp, 256]
            
            # B. MASK POOLING: Chỉ lấy đặc trưng tại vùng có xe (Dựa trên GT Heatmap)
            # Đây là mấu chốt: Tách chiếc xe ra khỏi bối cảnh nền
            v_target_temporal = (feats * t_hm_mask).sum(dim=(2, 3)) / (t_hm_mask.sum(dim=(2, 3)) + 1e-6) # [B, T, 256]
            
            # C. Đẩy lên không gian CLIP và Chuẩn hóa
            v_proj = vlm_head.video_to_clip(v_target_temporal)
            v_proj = F.normalize(v_proj, p=2, dim=-1).float() # [B, T, 512]
            t_proj = F.normalize(t_tokens, p=2, dim=-1).float() # [B, 512]
            
            # C. Đẩy lên không gian CLIP và Chuẩn hóa
            v_proj = vlm_head.video_to_clip(v_target_temporal)
            v_proj = F.normalize(v_proj, p=2, dim=-1).float()   # [B, T, 512]
            t_proj = F.normalize(t_tokens, p=2, dim=-1).float() # [B, N, 512] (với N thường là 32)
            
            # D. TÍNH MA TRẬN TƯƠNG ĐỒNG (Giống hệt hàm Train)
            # Nhân chéo mọi Video Frame (T) với mọi Text Token (N) của toàn bộ Batch (B)
            # Kết quả sim có shape: [B_video, B_text, T, N]
            sim = torch.einsum('vtd,bnd->vbtn', v_proj, t_proj) 
            
            # Tính điểm cuối cùng: Lấy Token có điểm cao nhất (max), rồi lấy trung bình qua các Frame (mean)
            # Kết quả sim_matrix có shape: [B_video, B_text] -> Ma trận vuông [B, B]
            sim_matrix = sim.max(dim=3)[0].mean(dim=2)

    # ==========================================
    # 4. HIỂN THỊ KẾT QUẢ ĐÁNH GIÁ TRỰC QUAN
    # ==========================================
    print("\n" + "="*50)
    print("📊 MA TRẬN ĐỘ TƯƠNG ĐỒNG (COSINE SIMILARITY)")
    print("="*50)
    print("Dòng = Video (vùng có xe), Cột = Text mô tả")
    print("Mục tiêu: Đường chéo chính (V1-T1, V2-T2...) phải có điểm CAO NHẤT.\n")
    
    sim_matrix = sim_matrix.cpu().numpy()
    
    # In Header
    header = f"{'':<10}" + "".join([f"Text {i+1:<8}" for i in range(B)])
    print(header)
    print("-" * (10 + 9*B))
    
    # In từng dòng ma trận
    for i in range(B):
        row_str = f"Video {i+1:<4}| "
        for j in range(B):
            score = sim_matrix[i, j]
            # Đánh dấu sao cho Cặp Đúng (Positive Pair)
            if i == j:
                row_str += f"*{score:.3f}* " 
            else:
                row_str += f"{score:.3f}   "
        print(row_str)
        
    print("\n📌 KẾT LUẬN NHANH:")
    for i in range(B):
        correct_score = sim_matrix[i, i]
        other_scores = [sim_matrix[i, j] for j in range(B) if j != i]
        max_other = max(other_scores) if other_scores else 0
        
        if correct_score > max_other:
            print(f"✅ Video {i+1}: Nhận diện NGỮ NGHĨA TỐT (Điểm đúng {correct_score:.3f} > Điểm sai cao nhất {max_other:.3f})")
        else:
            print(f"⚠️ Video {i+1}: Bị nhầm lẫn! (Điểm đúng {correct_score:.3f} < Điểm Text sai {max_other:.3f})")

if __name__ == "__main__":
    # Đảm bảo đường dẫn tới checkpoint và data của bạn là chính xác
    test_semantic_alignment(
        checkpoint_path="./checkpoints/limavlm_best.pth", 
        data_root="./data/data", 
        json_path="./data/data/train-tracks.json"
    )