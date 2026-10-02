"""Controlled central-supervision context ablation and pretrained HRNet-W18."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import NativePatchDataset
from .research_variants import ResearchUNet

ROOT=Path(__file__).resolve().parents[1]


class ContextDataset(NativePatchDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.padded=[np.pad(im,112,mode='reflect') for im in self.images]

    def __getitem__(self,index):
        # Preserve the exact random draw sequence and target crops of NativePatchDataset.
        if torch.rand(()) < self.enriched_probability:
            sources=tuple(self.enriched)
            source=sources[int(torch.randint(len(sources),()).item())]
            tiles=self.enriched[source]
        else:
            source=int(torch.randint(len(self.paths),()).item())
            tiles=self.origins[source]
        y,x=tiles[int(torch.randint(len(tiles),()).item())]
        image=torch.from_numpy(self.padded[source][y:y+448,x:x+448].copy()).float()[None]/255.
        target=torch.from_numpy(self.labels[source][y:y+224,x:x+224].astype(np.int64,copy=True))
        k=int(torch.randint(4,()).item())
        image,target=image.rot90(k,(-2,-1)),target.rot90(k,(-2,-1))
        if bool(torch.randint(2,()).item()):image,target=image.flip(-1),target.flip(-1)
        return image,target,source


class ContextUNet(ResearchUNet):
    def __init__(self,pretrained=True):
        super().__init__('baseline',pretrained)

    def forward(self,x,auxiliary=False):
        logits,features=super().forward(x,auxiliary=True)
        if self.training:
            assert x.shape[-2:]==(448,448)
            logits=logits[...,112:336,112:336]
            features=features[...,112:336,112:336]
        return (logits,features) if auxiliary else logits


class HRNetSegmenter(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        sys.path.insert(0,str(ROOT/'.vendor'))
        import timm
        self.encoder=timm.create_model('hrnet_w18',pretrained=False,features_only=True,
                                       feature_location='',out_indices=(1,2,3,4),in_chans=1)
        if pretrained:
            from safetensors.torch import load_file
            state=load_file(str(ROOT/'.torch_cache/hrnet/hrnet_w18.ms_in1k.safetensors'))
            state['conv1.weight']=state['conv1.weight'].sum(1,keepdim=True)
            expected=self.encoder.state_dict()
            selected={k:v for k,v in state.items() if k in expected}
            self.encoder.load_state_dict(selected,strict=True)
        self.head=nn.Sequential(nn.Conv2d(270,128,1,bias=False),nn.BatchNorm2d(128),nn.ReLU(),nn.Conv2d(128,4,1))
        self.register_buffer('input_mean',torch.tensor(.674325).view(1,1,1,1))
        self.register_buffer('input_std',torch.tensor(.229455).view(1,1,1,1))

    def forward(self,x,auxiliary=False):
        fs=self.encoder((x-self.input_mean)/self.input_std)
        merged=torch.cat([fs[0]]+[F.interpolate(f,size=fs[0].shape[-2:],mode='bilinear',align_corners=False) for f in fs[1:]],1)
        logits=F.interpolate(self.head(merged),size=x.shape[-2:],mode='bilinear',align_corners=False)
        return (logits,merged) if auxiliary else logits


def build_model(method,pretrained=True):
    if method=='context448':return ContextUNet(pretrained)
    if method=='hrnet_w18':return HRNetSegmenter(pretrained)
    raise ValueError(method)
