import torch
import torch.nn as nn
import torch.nn.functional as F

class DualPathways(nn.Module):
    def __init__(self):
        super().__init__()
        # Luồng Magno: Xử lý thô, phát hiện chuyển động nhanh
        self.magno_path = nn.Conv3d(3, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2))
        # Luồng Parvo: Xử lý chi tiết, chậm hơn
        self.parvo_path = nn.Conv3d(3, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1))

    def forward(self, video_stream):
        magno_features = self.magno_path(video_stream)
        parvo_features = self.parvo_path(video_stream)
        return magno_features, parvo_features