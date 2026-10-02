"""Last-stage upsampling ablations; DySample LP, scale 2, four groups.

Sampling follows https://github.com/tiny-smart/dysample (ICCV 2023),
with explicit bilinear gathers for MPS backward compatibility.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .resnet34_unet import ResNet34UNet


def sample_pixels(x, coords):
    n, c, h, w = x.shape
    oh, ow = coords.shape[1:3]
    xx, yy = coords[..., 0].clamp(0, w-1), coords[..., 1].clamp(0, h-1)
    x0, y0 = xx.floor().long(), yy.floor().long()
    x1, y1 = (x0+1).clamp(max=w-1), (y0+1).clamp(max=h-1)
    wx, wy = (xx-x0)[:, None], (yy-y0)[:, None]
    def at(y, z):
        indices = (y*w+z).reshape(n, 1, -1).expand(-1, c, -1)
        return x.flatten(2).gather(2, indices).reshape(n, c, oh, ow)
    return (1-wx)*(1-wy)*at(y0,x0)+wx*(1-wy)*at(y0,x1)+(1-wx)*wy*at(y1,x0)+wx*wy*at(y1,x1)


class DySampleLP(nn.Module):
    def __init__(self, channels=64, groups=4):
        super().__init__()
        self.groups = groups
        self.offset = nn.Conv2d(channels, 8*groups, 1)
        nn.init.normal_(self.offset.weight, std=.001)
        nn.init.zeros_(self.offset.bias)
        yy, xx = torch.meshgrid(torch.tensor([-.25,.25]), torch.tensor([-.25,.25]), indexing='ij')
        self.register_buffer('positions', torch.stack([xx,yy]).reshape(2,1,4,1,1).repeat(1,groups,1,1,1).reshape(1,8*groups,1,1))

    def coordinates(self, x):
        b, _, h, w = x.shape
        offsets = (.25*self.offset(x)+self.positions).reshape(b,2,4*self.groups,h,w)
        yy, xx = torch.meshgrid(torch.arange(h,device=x.device,dtype=x.dtype), torch.arange(w,device=x.device,dtype=x.dtype), indexing='ij')
        points = offsets + torch.stack([xx,yy])[None,:,None]
        return F.pixel_shuffle(points.reshape(b,-1,h,w),2).reshape(b,2,self.groups,2*h,2*w).permute(0,2,3,4,1).reshape(b*self.groups,2*h,2*w,2)

    def forward(self, x):
        b,c,h,w = x.shape
        return sample_pixels(x.reshape(b*self.groups,c//self.groups,h,w),self.coordinates(x)).reshape(b,c,2*h,2*w)


class UpsamplingUNet(ResNet34UNet):
    def __init__(self, method, pretrained=True):
        super().__init__(pretrained=pretrained, use_scse=True)
        self.method = method
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1701)
            if method == 'fullres_skip':
                self.detail = nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU())
                old = self.decoder0.block[0]
                expanded = nn.Conv2d(80,32,3,padding=1,bias=False)
                with torch.no_grad():
                    expanded.weight[:,:64].copy_(old.weight)
                    expanded.weight[:,64:].zero_()
                self.decoder0.block[0] = expanded
            elif method == 'pixelshuffle':
                conv = nn.Conv2d(64,256,3,padding=1)
                # Repeated subpixel kernels start without phase-specific differences.
                with torch.no_grad():
                    kernel = torch.empty(64,64,3,3)
                    nn.init.kaiming_normal_(kernel)
                    conv.weight.copy_(kernel.repeat_interleave(4,dim=0))
                    conv.bias.zero_()
                self.upsampler = nn.Sequential(conv,nn.PixelShuffle(2))
            elif method == 'dysample':
                self.upsampler = DySampleLP()
            else:
                raise ValueError(method)

    def forward(self, x, auxiliary=False):
        size = x.shape[-2:]
        raw = (x-self.input_mean)/self.input_std if self.pretrained else x
        stem = self.relu(self.bn1(self.conv1(raw)))
        e1 = self.layer1(self.maxpool(stem))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        x = self.layer4(e3)
        for skip, decoder in ((e3,self.decoder4),(e2,self.decoder3),(e1,self.decoder2),(stem,self.decoder1)):
            x = decoder(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
        if self.method == 'fullres_skip':
            x = torch.cat([F.interpolate(x,size=size,mode='bilinear',align_corners=False),self.detail(raw)],1)
        else:
            x = self.upsampler(x)
            if x.shape[-2:] != size:
                x = x[...,:size[0],:size[1]]
        features = self.decoder0(x)
        logits = self.head(features)
        return (logits,features) if auxiliary else logits
