import torch
import torch.nn as nn
import torch.nn.functional as F

class LiMaLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_cm=1.0, lambda_temp=0.5):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        self.lambda_cm = lambda_cm
        self.lambda_temp = lambda_temp

    def cross_modal_loss(self, video_feat, text_feat, is_match):
        B = video_feat.size(0)
        if B < 2: return torch.tensor(0.0, device=video_feat.device)

        video_feat = F.normalize(video_feat, dim=1, eps=1e-8)
        text_feat = F.normalize(text_feat, dim=1, eps=1e-8)

        scale = torch.clamp(self.logit_scale.exp(), max=100)
        logits = torch.matmul(video_feat, text_feat.T) * scale

        labels = torch.arange(B, device=video_feat.device)
        loss_v2t = F.cross_entropy(logits, labels, reduction='none')
        loss_t2v = F.cross_entropy(logits.T, labels, reduction='none')

        # Chỉ phạt trên các mẫu Positive
        valid_pairs_count = is_match.sum()
        if valid_pairs_count < 1: return torch.tensor(0.0, device=video_feat.device)

        return ((loss_v2t + loss_t2v) / 2.0 * is_match).sum() / (valid_pairs_count + 1e-8)

    def temporal_loss(self, feat_seq, mask_5d=None):
        if feat_seq is None or feat_seq.dim() != 3: return torch.tensor(0.0, device=feat_seq.device)
        B, T, D = feat_seq.shape
        if T < 2: return torch.tensor(0.0, device=feat_seq.device)

        if mask_5d is not None and mask_5d.dim() == 5:
            valid_mask = (mask_5d.mean(dim=[1, 3, 4]) > 0.0).float()
            temporal_mask = valid_mask[:, 1:] * valid_mask[:, :-1]
        else:
            temporal_mask = torch.ones(B, T - 1, device=feat_seq.device)

        valid_count = temporal_mask.sum()
        if valid_count < 1: return torch.tensor(0.0, device=feat_seq.device)

        cos_sim = F.cosine_similarity(feat_seq[:, 1:], feat_seq[:, :-1], dim=-1)
        return ((1.0 - cos_sim) * temporal_mask).sum() / (valid_count + 1e-8)

    def forward(self, video_feat, text_feat, feat_seq, is_match, mask=None):
        L_cm = self.cross_modal_loss(video_feat, text_feat, is_match)
        L_temp = self.temporal_loss(feat_seq, mask)
        return {"loss": (self.lambda_cm * L_cm) + (self.lambda_temp * L_temp), "L_cm": L_cm, "L_temp": L_temp}