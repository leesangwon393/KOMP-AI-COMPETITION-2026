"""RSU backbone and multiclass full U²-Net adapters.

Based on xuebinqin/U-2-Net (Apache-2.0). Modified design: backbone feature
adapters, optional scSE, raw multiclass logits, center-only supervision.
The unchanged upstream blocks are in vendor_u2net.py; see LICENSE_U2NET.txt.
"""
import torch
from torch import nn
from torch.nn import functional as F

from . import vendor_u2net as upstream


def resize(x,size):
    return F.interpolate(x,size=size,mode='bilinear',align_corners=False)


class RSUEncoder(nn.Module):
    channels=[64,128,256,512,512,512]

    def __init__(self):
        super().__init__()
        self.stage1=upstream.RSU7(3,32,64)
        self.stage2=upstream.RSU6(64,32,128)
        self.stage3=upstream.RSU5(128,64,256)
        self.stage4=upstream.RSU4(256,128,512)
        self.stage5=upstream.RSU4F(512,256,512)
        self.stage6=upstream.RSU4F(512,256,512)
        self.pool=nn.MaxPool2d(2,stride=2,ceil_mode=True)

    def forward(self,x):
        result=[]
        for i in range(1,7):
            x=getattr(self,f'stage{i}')(x)
            result.append(x)
            if i<6: x=self.pool(x)
        return result


class RSUBackbone(nn.Module):
    """Raw stages2–6 at OS2/4/8/16/32, adapted to B's five skip widths."""
    channels=[64,256,512,1024,2048]

    def __init__(self,mean,std):
        super().__init__()
        self.raw=RSUEncoder()
        self.adapters=nn.ModuleList([
            nn.Sequential(nn.Conv2d(cin,cout,1,bias=False),nn.BatchNorm2d(cout),nn.ReLU())
            if cin!=cout else nn.Identity()
            for cin,cout in zip(self.raw.channels[1:],self.channels)])
        self.register_buffer('input_mean',torch.tensor(mean).reshape(1,3,1,1))
        self.register_buffer('input_std',torch.tensor(std).reshape(1,3,1,1))

    def forward(self,x):
        if x.shape[1]!=1: raise ValueError('Expected grayscale input')
        z=(x.repeat(1,3,1,1)-self.input_mean)/self.input_std
        raw=self.raw(z)
        return [adapter(feature) for adapter,feature in zip(self.adapters,raw[1:])]


class FullU2Net(upstream.U2NET):
    """Keep upstream feature widths/6-head fusion; return logits, never sigmoid.

    Optional attention is after each of five decoder RSUs. New gates are
    initialized under an isolated RNG, preserving the upstream allocation.
    """
    def __init__(self,scse_factory,scse,seed):
        super().__init__(in_ch=3,out_ch=4)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+4101)
            gates=[scse_factory(c) for c in (512,256,128,64,64)]
        self.attentions=nn.ModuleList(gates if scse else [nn.Identity() for _ in gates])

    def forward(self,x):
        encoder=[]
        z=x
        for i in range(1,7):
            z=getattr(self,f'stage{i}')(z);encoder.append(z)
            if i<6: z=getattr(self,f'pool{i}{i+1}')(z)
        decoded=[]
        for index,i in enumerate(range(5,0,-1)):
            skip=encoder[i-1]
            z=getattr(self,f'stage{i}d')(torch.cat([resize(z,skip.shape[-2:]),skip],1))
            z=self.attentions[index](z);decoded.append(z)
        features=list(reversed(decoded))+[encoder[-1]]
        sides=[resize(getattr(self,f'side{i}')(feature),x.shape[-2:])
               for i,feature in enumerate(features,1)]
        fused=self.outconv(torch.cat(sides,1))
        return fused,sides
