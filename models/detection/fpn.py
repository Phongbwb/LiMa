import torch
import torch.nn as nn
import torch.nn.functional as F

class TrueFPN(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.out_channels = out_channels

        # Lateral Conv: Chuyển đổi channel
        self.lat3 = nn.Conv2d(in_channels, out_channels, 1)
        self.lat4 = nn.Conv2d(in_channels, out_channels, 1)
        self.lat5 = nn.Conv2d(in_channels, out_channels, 1)

        # Smooth Conv: Chống nhiễu (aliasing) sau khi cộng gộp Upsample
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        """
        x: [B*T, C, H, W] (Cần đảm bảo H, W đủ lớn, vd: từ 28x28 trở lên)
        """
        # 1. Bottom-up (Tạo tháp kích thước)
        c3 = x
        c4 = F.max_pool2d(x, 2) # Dùng MaxPool để giữ lại đặc trưng viền cạnh tốt hơn AvgPool
        c5 = F.max_pool2d(x, 4)

        # 2. Lateral
        p5 = self.lat5(c5)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3)

        # 3. Top-down Pathway (LINH HỒN CỦA FPN)
        # Cộng gộp P5 vào P4
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        # Cộng gộp P4 vào P3
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")

        # 4. Smoothing
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)

        return [p3, p4, p5]

class YOLOHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=1, num_anchors=1):
        super().__init__()
        
        # Out channels = Số anchor * (x, y, w, h, conf + classes)
        out_dim = num_anchors * (5 + num_classes)
        
        self.head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels), # Thêm BN để hội tụ nhanh hơn
                nn.SiLU(inplace=True),       # SiLU tốt hơn ReLU cho task Detection
                nn.Conv2d(in_channels, out_dim, 1)
            )
            for _ in range(3)
        ])

    def forward(self, features):
        """
        features: list [P3, P4, P5]
        """
        outputs = []
        for i, f in enumerate(features):
            outputs.append(self.head[i](f))
            
        return outputs