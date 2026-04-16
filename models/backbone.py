import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import math

class DeformablePatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=16):
        super().__init__()
        self.stride = patch_size
        self.kernel_size = patch_size + 3
        self.padding = self.kernel_size // 2
        self.pre_norm = nn.GroupNorm(1, in_channels)
        
        self.offset_net = nn.Conv2d(in_channels, 2 * self.kernel_size**2, 
                                   kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        nn.init.constant_(self.offset_net.weight, 0.)
        nn.init.constant_(self.offset_net.bias, 0.)

        self.weight = nn.Parameter(torch.Tensor(embed_dim, in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        x = self.pre_norm(x)
        offsets = torch.tanh(self.offset_net(x)) * 2.0
        return deform_conv2d(x, offsets, self.weight, self.bias, stride=self.stride, padding=self.padding)

class CustomVideoBackbone(nn.Module):
    def __init__(self, d_model=256, num_frames=8, img_size=384, patch_size=16):
        super().__init__()
        self.patch_embed = DeformablePatchEmbedding(3, d_model, patch_size)
        self.dw_conv3d = nn.Conv3d(d_model, d_model, kernel_size=(3, 3, 3), padding=(0, 1, 1), groups=d_model)
        
        self.temp_embed = nn.Parameter(torch.zeros(1, d_model, num_frames))
        self.spatial_embed = nn.Parameter(torch.zeros(1, d_model, img_size//patch_size, img_size//patch_size))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_2d = x.transpose(1, 2).reshape(B * T, C, H, W)
        features_2d = self.patch_embed(x_2d)
        _, E, Hp, Wp = features_2d.shape
        
        patches_3d = features_2d.view(B, T, E, Hp, Wp).transpose(1, 2)
        patches_pad = F.pad(patches_3d, (0, 0, 0, 0, 2, 0)) 
        features_3d = F.gelu(self.dw_conv3d(patches_pad)) + patches_3d

        temp_pe = F.interpolate(self.temp_embed, size=T, mode='linear').transpose(1, 2).view(1, T, 1, 1, E)
        spatial_pe = F.interpolate(self.spatial_embed, size=(Hp, Wp), mode='bilinear').permute(0, 2, 3, 1).view(1, 1, Hp, Wp, E)
        
        out = features_3d.permute(0, 2, 3, 4, 1) + temp_pe + spatial_pe
        return self.norm(out) # [B, T, Hp, Wp, E]