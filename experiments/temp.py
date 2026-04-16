import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import deform_conv2d
import torchvision.transforms as T
from PIL import Image
from einops import rearrange
from tqdm import tqdm

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# ==========================================
# BACKBONE
# ==========================================
class DeformablePatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=8):
        super().__init__()
        self.stride = patch_size
        self.kernel_size = patch_size + 3
        self.padding = self.kernel_size // 2

        self.norm = nn.GroupNorm(1, in_channels)

        self.offset = nn.Conv2d(
            in_channels, 2*self.kernel_size**2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        nn.init.constant_(self.offset.weight, 0.)
        nn.init.constant_(self.offset.bias, 0.)

        self.weight = nn.Parameter(torch.randn(embed_dim, in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        x = self.norm(x)
        off = torch.tanh(self.offset(x)) * 2
        return deform_conv2d(x, off, self.weight, self.bias,
                             stride=self.stride, padding=self.padding)


class Backbone(nn.Module):
    def __init__(self, d=256, T=8):
        super().__init__()
        self.patch = DeformablePatchEmbedding(3, d)
        self.conv3d = nn.Conv3d(d, d, (3,3,3), padding=(0,1,1), groups=d)
        self.temp = nn.Parameter(torch.zeros(1,d,T))
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B,C,T,H,W = x.shape

        x2d = x.transpose(1,2).reshape(B*T,C,H,W)
        f = self.patch(x2d)
        _,D,H,W = f.shape

        f = f.view(B,T,D,H,W).transpose(1,2)

        f_pad = F.pad(f,(0,0,0,0,2,0))
        f = F.gelu(self.conv3d(f_pad)) + f

        temp = F.interpolate(self.temp,size=T).transpose(1,2).view(1,T,1,1,D)

        return self.norm(f.permute(0,2,3,4,1)+temp)


# ==========================================
# MAMBA BLOCK
# ==========================================
class Block(nn.Module):
    def __init__(self, d=256, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d,d*2)
        self.conv = nn.Conv1d(d,d,3,padding=2,groups=d)

        self.x_proj = nn.Linear(d,d+2*d_state)

        self.A_log = nn.Parameter(torch.log(torch.arange(1,d_state+1).float().repeat(d,1)))
        self.D = nn.Parameter(torch.ones(d))

        self.video_to_clip = nn.Linear(d,512)
        self.logit_scale = nn.Parameter(torch.ones([])*np.log(1/0.07))
        self.bias = nn.Parameter(torch.tensor(-1.0))

        self.out = nn.Linear(d,d)

    def forward(self,x,text,temp):
        B,L,D = x.shape

        xz = self.in_proj(self.norm(x))
        x,z = xz.chunk(2,-1)

        x = F.silu(self.conv(x.transpose(1,2))[:,:,:L]).transpose(1,2)

        v = F.normalize(self.video_to_clip(x),dim=-1,eps=1e-6)
        t = F.normalize(text,dim=-1,eps=1e-6).unsqueeze(1)

        sim = (v*t).sum(-1,keepdim=True)
        sim = torch.clamp(sim,-5,5)

        scale = torch.clamp(self.logit_scale.exp(),max=10)

        gate = torch.sigmoid((sim*scale+self.bias)/temp)
        gate = torch.nan_to_num(gate,0.5)

        proj = self.x_proj(x)
        delta,Bm,Cm = torch.split(proj,[D,16,16],dim=-1)

        A = -torch.exp(self.A_log)

        y = selective_scan_fn(
            x.transpose(1,2),
            delta.transpose(1,2),
            A,
            (Bm*gate).transpose(1,2),
            (Cm*gate).transpose(1,2),
            self.D,
            delta_softplus=True
        ).transpose(1,2)

        return self.out(y*F.silu(z))+x, gate, v


# ==========================================
# MODEL
# ==========================================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(4)])

        self.hm = nn.Sequential(nn.Conv2d(256,128,3,padding=1), nn.ReLU(), nn.Conv2d(128,1,1), nn.Sigmoid())
        self.sz = nn.Sequential(nn.Conv2d(256,128,3,padding=1), nn.ReLU(), nn.Conv2d(128,2,1))
        self.off = nn.Sequential(nn.Conv2d(256,128,3,padding=1), nn.ReLU(), nn.Conv2d(128,2,1))

    def forward(self,x,text,temp):
        B,T,H,W,D = x.shape
        x = rearrange(x,"b t h w d -> b (t h w) d")

        gates=[]
        video_embs=[]

        for b in self.blocks:
            x,g,v = b(x,text,temp)
            gates.append(g)
            video_embs.append(v)

        v_final = video_embs[-1].mean(1)

        x2d = rearrange(x,"b (t h w) d -> (b t) d h w",t=T,h=H,w=W)

        return self.hm(x2d),self.sz(x2d),self.off(x2d),gates,v_final


# ==========================================
# DATASET (FIXED)
# ==========================================
class DatasetCF(Dataset):
    def __init__(self, root, json_path, img_size=384, down_ratio=8):
        self.root=root
        self.img_size=img_size
        self.down_ratio=down_ratio

        with open(json_path) as f:
            data=json.load(f)

        self.samples=[]
        for tid,info in data.items():
            for t in info["nl"]:
                self.samples.append((info,t))

        self.text=torch.load(os.path.join(root,"clip_text_embeddings.pt"))

        self.tf=T.Compose([
            T.Resize((img_size,img_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3,[0.5]*3)
        ])

    def __len__(self): return len(self.samples)

    def draw_gaussian(self,hmap,cx,cy,r=2):
        for dx in range(-r,r+1):
            for dy in range(-r,r+1):
                xx=int(cx+dx)
                yy=int(cy+dy)
                if 0<=xx<hmap.shape[1] and 0<=yy<hmap.shape[0]:
                    val=np.exp(-(dx*dx+dy*dy)/2)
                    hmap[yy,xx]=max(hmap[yy,xx],val)

    def __getitem__(self, idx):
        s = self.samples[idx]

        text = s["text"].strip().lower()
        if text in self.text_embs:
            text_emb = self.text_embs[text]
        else:
            text_emb = torch.zeros(512)  # ❗ không dùng random

        indices = np.linspace(0, len(s["frames"])-1, self.num_frames, dtype=int)
        hms = self.img_size // self.down_ratio

        video, hm, sz, off = [], [], [], []

        for i in indices:
            img = Image.open(os.path.join(self.data_root, s["frames"][i])).convert('RGB')
            w_orig, h_orig = img.size
            video.append(self.transform(img))

            hmap = np.zeros((hms, hms), dtype=np.float32)
            smap = np.zeros((2, hms, hms), dtype=np.float32)
            omap = np.zeros((2, hms, hms), dtype=np.float32)

            box = s["boxes"][i]
            cx = (box[0] + box[2]/2) / w_orig * hms
            cy = (box[1] + box[3]/2) / h_orig * hms
            bw = box[2] / w_orig * hms
            bh = box[3] / h_orig * hms

            radius = max(1, int(math.sqrt(bw * bh) * 0.2))
            self._draw_gaussian(hmap, (cx, cy), radius)

            xi, yi = int(cx), int(cy)
            if 0 <= xi < hms and 0 <= yi < hms:
                smap[:, yi, xi] = np.log(np.array([bw, bh], dtype=np.float32) + 1e-6)
                omap[:, yi, xi] = np.array([cx - xi, cy - yi], dtype=np.float32)

            hm.append(torch.from_numpy(hmap).unsqueeze(0))
            sz.append(torch.from_numpy(smap))
            off.append(torch.from_numpy(omap))

        return {
            "video": torch.stack(video, dim=1),
            "text_emb": text_emb.float(),
            "hm": torch.stack(hm),
            "sz": torch.stack(sz),
            "off": torch.stack(off)
        }

# ==========================================
# LOSSES
# ==========================================
def masked_l1_loss(pred, target, hm):
    mask = (hm > 0.3).expand_as(pred)
    if mask.sum() == 0:
        return torch.tensor(0., device=pred.device)
    return F.l1_loss(pred[mask], target[mask])

def focal(pred, gt):
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()

    neg_weights = (1 - gt) ** 4

    pred = pred.clamp(1e-6, 1-1e-6)

    pos_loss = -torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = -torch.log(1 - pred) * pred ** 2 * neg * neg_weights

    num_pos = pos.sum()

    if num_pos == 0:
        return neg_loss.sum()
    else:
        return (pos_loss.sum() + neg_loss.sum()) / num_pos


def reg(pred, gt, hm):
    mask = (hm > 0.1).expand_as(pred)
    if mask.sum()==0:
        return torch.tensor(0.,device=pred.device)
    return F.l1_loss(pred[mask],gt[mask])


def clip_loss(v,t,scale):
    v=F.normalize(v,dim=-1)
    t=F.normalize(t,dim=-1)

    logits=v@t.t()*scale
    labels=torch.arange(v.size(0)).to(v.device)

    return (F.cross_entropy(logits,labels)+F.cross_entropy(logits.t(),labels))/2


def gate_reg(gates):
    g=torch.cat(gates,1)
    entropy=-(g*torch.log(g+1e-6)+(1-g)*torch.log(1-g+1e-6)).mean()
    var=g.var()
    return -entropy + 0.1*(1/(var+1e-6))


# ==========================================
# TRAIN
# ==========================================
def train():
    device="cuda"

    backbone=Backbone().to(device)
    model=Model().to(device)

    opt=torch.optim.AdamW(list(backbone.parameters())+list(model.parameters()),lr=1e-4)

    ds=DatasetCF("./data/data","./data/data/train-tracks.json")
    dl=DataLoader(ds,4,True,num_workers=4)

    os.makedirs("ckpt",exist_ok=True)
    best=1e9

    for epoch in range(20):
        temp=max(0.5,1-epoch*0.03)

        total=0

        for batch in tqdm(dl):
            v=batch["video"].to(device)
            t=batch["text"].to(device)

            hm=batch["hm"].view(-1,1,48,48).to(device)
            sz=batch["sz"].view(-1,2,48,48).to(device)
            off=batch["off"].view(-1,2,48,48).to(device)

            opt.zero_grad()

            f=backbone(v)
            ph,ps,po,gates,v_emb=model(f,t,temp)

            l1=focal(ph,hm)
            l2=reg(ps,sz,hm)
            l3=reg(po,off,hm)

            det=l1+0.1*l2+0.5*l3

            scale=torch.clamp(model.blocks[0].logit_scale.exp(),max=10)
            c_loss=clip_loss(v_emb,t,scale)

            g_loss=gate_reg(gates)

            loss=det+0.1*c_loss+0.01*g_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()

            total+=loss.item()

        total/=len(dl)

        if total<best:
            best=total
            torch.save({
                "backbone":backbone.state_dict(),
                "model":model.state_dict()
            },"ckpt/best.pth")
            print("🔥 BEST:",best)

        print(f"Epoch {epoch} Loss {total:.4f}")


if __name__=="__main__":
    train()