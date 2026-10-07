"""Independent paper-inspired modules; exact paper reproductions are not claimed."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .baseline_models import resize, SCSE
from .followup_modules import FrequencyFusionNoOffset

class ECA(nn.Module):
    def __init__(self,c):
        super().__init__();self.conv=nn.Conv1d(1,1,3,padding=1,bias=False)
    def forward(self,x):
        gate=self.conv(x.mean((-2,-1))[:,None]).sigmoid()[:,0,:,None,None]
        return x*gate

class CBAM(nn.Module):
    def __init__(self,c):
        super().__init__();self.channel=nn.Sequential(nn.Conv2d(c,max(c//16,1),1),nn.ReLU(),nn.Conv2d(max(c//16,1),c,1))
        self.spatial=nn.Conv2d(2,1,7,padding=3)
    def forward(self,x):
        x=x*(self.channel(x.mean((-2,-1),keepdim=True))+self.channel(x.amax((-2,-1),keepdim=True))).sigmoid()
        return x*self.spatial(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1)).sigmoid()

class APAM(nn.Module):
    def __init__(self,c):
        super().__init__();self.channel=ECA(c);self.spatial=nn.Conv2d(c,1,1);self.weight=nn.Parameter(torch.zeros(2))
    def forward(self,x):
        w=self.weight.softmax(0);return w[0]*self.channel(x)+w[1]*x*self.spatial(x).sigmoid()

def attention(c,kind):
    return {'scse':lambda:SCSE(c),'none':nn.Identity,'eca':lambda:ECA(c),'cbam':lambda:CBAM(c),
            'eca_cbam':lambda:nn.Sequential(ECA(c),CBAM(c)),'apam':lambda:APAM(c)}[kind]()

class MultiDilation(nn.Module):
    def __init__(self,c):
        super().__init__();self.branches=nn.ModuleList([nn.Sequential(nn.Conv2d(c,c,3,padding=d,dilation=d,groups=c),nn.ReLU()) for d in (1,2,4)])
        self.weights=nn.Parameter(torch.zeros(3));self.project=nn.Conv2d(c,c,1)
    def forward(self,x):return x+self.project(sum(w*b(x) for w,b in zip(self.weights.softmax(0),self.branches)))

class ChannelLSTM(nn.Module):
    def __init__(self,c):
        super().__init__();self.rnn=nn.LSTM(1,8,batch_first=True,bidirectional=True);self.output=nn.Linear(16,1)
    def forward(self,x):
        z,_=self.rnn(x.mean((-2,-1)).unsqueeze(-1));g=self.output(z).sigmoid().transpose(1,2).unsqueeze(-1)
        return x*g.transpose(1,2)

class CompressedContext(nn.Module):
    def __init__(self,c,kind):
        super().__init__();self.down=nn.Conv2d(c,64,1);self.up=nn.Conv2d(64,c,1)
        self.core=nn.Sequential(*([MultiDilation(64)] if kind in ('aspp','cmaa') else []),
            *([APAM(64)] if kind=='cmaa' else []),*([ChannelLSTM(64)] if kind in ('lstm','cmaa') else []))
    def forward(self,x):return x+self.up(self.core(self.down(x)))

class OCR(nn.Module):
    """Four class prototypes, pixel-to-region attention and auxiliary region CE."""
    def __init__(self,c=32):
        super().__init__();self.coarse=nn.Conv2d(c,4,1);self.query=nn.Conv2d(c,c,1);self.key=nn.Linear(c,c);self.fuse=nn.Conv2d(2*c,c,1)
    def forward(self,x):
        b,c,h,w=x.shape;coarse=self.coarse(x)
        distribution=coarse.flatten(2).softmax(-1)
        prototypes=distribution@x.flatten(2).transpose(1,2)
        relation=(self.query(x).flatten(2).transpose(1,2)@self.key(prototypes).transpose(1,2)/math.sqrt(c)).softmax(-1)
        context=(relation@prototypes).transpose(1,2).reshape(b,c,h,w)
        return x+self.fuse(torch.cat([x,context],1)),coarse

class Snake(nn.Module):
    """Five-point accumulated-offset axial sampling; no persistent-homology loss."""
    def __init__(self,c=32):
        super().__init__();self.offset=nn.Conv2d(c,8,3,padding=1)
        self.axial=nn.ModuleList([nn.Conv2d(c,c,1) for _ in range(10)])
        self.fuse=nn.Conv2d(3*c,c,1);self.standard=nn.Conv2d(c,c,3,padding=1)
    def forward(self,x):
        b,c,h,w=x.shape;yy,xx=torch.meshgrid(torch.arange(h,device=x.device,dtype=x.dtype),torch.arange(w,device=x.device,dtype=x.dtype),indexing='ij')
        offsets=self.offset(x).tanh().reshape(b,2,4,h,w);outputs=[]
        for axis in range(2):
            cur=offsets[:,axis];neg=cur[:,:2].flip(1).cumsum(1).flip(1);pos=cur[:,2:].cumsum(1)
            bends=torch.cat([neg,torch.zeros_like(cur[:,:1]),pos],1)
            samples=[]
            for k in range(5):
                dx=(k-2) if axis==0 else bends[:,k];dy=bends[:,k] if axis==0 else (k-2)
                gx=2*(xx[None]+dx)/max(w-1,1)-1;gy=2*(yy[None]+dy)/max(h-1,1)-1
                gx=gx.expand(b,h,w);gy=gy.expand(b,h,w)
                z=F.grid_sample(x,torch.stack([gx,gy],-1),align_corners=True,padding_mode='border')
                samples.append(self.axial[axis*5+k](z))
            outputs.append(sum(samples)/5)
        return x+self.fuse(torch.cat([self.standard(x),*outputs],1))

class FrequencyDetail(nn.Module):
    def __init__(self,c):
        super().__init__();self.low=nn.Conv2d(c,c,1);self.high=nn.Conv2d(c,c,1);self.gate=SCSE(c)
    def forward(self,x):
        low=F.avg_pool2d(x,3,stride=1,padding=1,count_include_pad=False)
        return self.gate(self.low(low)+self.high(x-low))

class WeightedSkip(nn.Module):
    def __init__(self,channels):
        super().__init__();self.gates=nn.ModuleList([nn.Conv2d(c,1,1) for c in channels]);self.weights=nn.Parameter(torch.zeros(len(channels)))
    def forward(self,features):return [x*(1+self.weights[i].sigmoid()*g(x).sigmoid()) for i,(x,g) in enumerate(zip(features,self.gates))]

class AdjacentCross(nn.Module):
    def __init__(self,channels):
        super().__init__();self.q=nn.ModuleList([nn.Conv2d(c,32,1) for c in channels[:-1]])
        self.kv=nn.ModuleList([nn.Conv2d(c,64,1) for c in channels[1:]]);self.out=nn.ModuleList([nn.Conv2d(32,c,1) for c in channels[:-1]])
    def forward(self,features):
        result=[]
        for x,y,q,kv,out in zip(features[:-1],features[1:],self.q,self.kv,self.out):
            b,_,h,w=x.shape;k,v=kv(F.adaptive_avg_pool2d(y,(7,7))).flatten(2).chunk(2,1)
            z=(q(x).flatten(2).transpose(1,2)@k/math.sqrt(32)).softmax(-1)@v.transpose(1,2)
            result.append(x+out(z.transpose(1,2).reshape(b,32,h,w)))
        return result+[features[-1]]

class OffsetFusion(nn.Module):
    def __init__(self):
        super().__init__();self.base=FrequencyFusionNoOffset();self.offset=nn.Conv2d(128,2,3,padding=1)
        nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)
    def forward(self,high,low):
        high,low=self.base(high,low);b,c,h,w=low.shape
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=low.device,dtype=low.dtype),torch.linspace(-1,1,w,device=low.device,dtype=low.dtype),indexing='ij')
        delta=self.offset(torch.cat([high,low],1)).tanh().permute(0,2,3,1)
        delta=delta*delta.new_tensor([2/max(w-1,1),2/max(h-1,1)])
        grid=torch.stack([xx,yy],-1)[None]+delta
        return high,F.grid_sample(low,grid,align_corners=True,padding_mode='border')
