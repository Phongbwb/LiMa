import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, alpha=0.5):
        """
        Hàm Loss Tương phản Phân cấp cho FGVC.
        Args:
            temperature (float): Hệ số nhiệt độ, càng nhỏ mạng càng khắt khe với các vector sai lệch.
            alpha (float): Trọng số cân bằng giữa [0, 1]. 
                           Ví dụ: 0.4 nghĩa là 40% tập trung phân loại kiểu dáng, 60% tập trung phân biệt hãng xe.
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, features, coarse_labels, fine_labels):
        """
        Args:
            features: [BatchSize, EmbedDim] - Vector đầu ra cuối cùng của Astrocytic Pool.
            coarse_labels: [BatchSize] - Nhãn loại xe (VD: 0=Scooter, 1=Underbone, 2=Sedan).
            fine_labels: [BatchSize] - Nhãn hãng/dòng xe (VD: 0=Honda Vision, 1=Yamaha Janus).
        """
        device = features.device
        batch_size = features.shape[0]

        # 0. Chuẩn hóa L2 (L2 Normalization)
        # Bắt buộc để biến khoảng cách Euclid thành Cosine Similarity
        features = F.normalize(features, p=2, dim=1)

        # 1. Tính Ma trận Tương đồng (Cosine Similarity Matrix)
        # Kết quả là ma trận [BatchSize, BatchSize]
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 2. Tạo Mặt nạ (Masks) cho các cấp độ
        # Coarse Mask: 1 nếu cùng loại xe, 0 nếu khác
        coarse_mask = torch.eq(coarse_labels.unsqueeze(1), coarse_labels.unsqueeze(0)).float().to(device)
        # Fine Mask: 1 nếu cùng hãng, 0 nếu khác
        fine_mask = torch.eq(fine_labels.unsqueeze(1), fine_labels.unsqueeze(0)).float().to(device)

        # Mask loại bỏ đường chéo chính (không cho phép 1 chiếc xe tự so sánh với chính nó)
        logits_mask = torch.scatter(
            torch.ones_like(coarse_mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        coarse_mask = coarse_mask * logits_mask
        fine_mask = fine_mask * logits_mask

        # ==========================================
        # 3. Tính Loss Cấp độ 1 (Coarse - Kiểu dáng)
        # ==========================================
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob_coarse = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        
        # Mean log-likelihood cho các cặp cùng kiểu dáng
        coarse_loss = - (coarse_mask * log_prob_coarse).sum(1) / (coarse_mask.sum(1) + 1e-8)
        coarse_loss = coarse_loss.mean()

        # ==========================================
        # 4. Tính Loss Cấp độ 2 (Fine - Hãng xe)
        # ==========================================
        # CHÌA KHÓA: Mẫu số của log_prob_fine CHỈ tính tổng các xe đã nằm trong cùng Cấp độ 1 (coarse_mask)
        # Điều này ép mạng chỉ phân biệt Honda/Yamaha TRONG NỘI BỘ nhóm Scooter
        log_prob_fine = sim_matrix - torch.log((exp_sim * coarse_mask).sum(1, keepdim=True) + 1e-8)
        
        fine_loss = - (fine_mask * log_prob_fine).sum(1) / (fine_mask.sum(1) + 1e-8)
        fine_loss = fine_loss.mean()

        # Tổng hợp Loss
        total_loss = self.alpha * coarse_loss + (1.0 - self.alpha) * fine_loss
        return total_loss