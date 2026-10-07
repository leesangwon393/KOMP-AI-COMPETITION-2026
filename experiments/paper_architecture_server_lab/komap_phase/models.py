"""Maintained B, pinned library architectures and explicitly labelled derived modules."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as tv
from .baseline_models import (SegmentationModel as Baseline, GrayEncoder, UNetDecoder, SCSE, ConvBlock, center_half, resize)
from .rsu_models import RSUBackbone, FullU2Net
from .vendor_u2net import U2NETP
from .paper_modules import (attention, CompressedContext, OCR, Snake, FrequencyDetail, WeightedSkip, AdjacentCross, OffsetFusion)
from .followup_modules import FrequencyFusionNoOffset, PixelShuffleUpsample

def transfer_micronet(model,kind):
    suffix='microscopynet' if kind=='micronet' else 'imagenet-microscopynet'
    url='https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/resnet101_pretrained_'+suffix+'_v1.0.pth.tar'
    cache=Path(torch.hub.get_dir())/'checkpoints';cache.mkdir(parents=True,exist_ok=True);path=cache/url.rsplit('/',1)[-1]
    if not path.exists():torch.hub.download_url_to_file(url,str(path))
    state=torch.load(path,map_location='cpu',weights_only=True)
    state=state.get('state_dict',state)
    state={k.removeprefix('module.'):v for k,v in state.items()}
    allowed={k:v for k,v in state.items() if not k.startswith('fc.')}
    expected={k:v for k,v in model.encoder.state_dict().items() if not k.startswith('input_')}
    if set(allowed)!=set(expected) or any(v.shape!=expected[k].shape for k,v in allowed.items()):
        raise ValueError('NASA checkpoint does not exactly match R101 encoder; transfer aborted')
    model.encoder.load_state_dict({**model.encoder.state_dict(),**allowed},strict=True)
    model.initialization_record={'kind':kind,'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'version':'1.0','normalization':'fixed ImageNet'}

class Encoder(nn.Module):
    """Return OS2/4/8/16/32; synthesize OS2 detail only if backbone lacks it."""
    def __init__(self,name,pretrained,config):
        super().__init__();self.timm=name.startswith('timm:')
        if self.timm:
            import timm
            options={'img_size':config['train']['patch']} if 'swin_' in name else {}
            self.raw=timm.create_model(name[5:],pretrained=pretrained,features_only=True,**options)
            channels=self.raw.feature_info.channels();reductions=self.raw.feature_info.reduction()
            self.indices=[next((j for j,r in enumerate(reductions) if r==s),None) for s in (2,4,8,16,32)]
            self.channels=[channels[i] if i is not None else 64 for i in self.indices]
        else:
            import segmentation_models_pytorch as smp
            self.raw=smp.encoders.get_encoder(name,in_channels=3,depth=5,weights='imagenet' if pretrained else None)
            channels=self.raw.out_channels;self.indices=[i if channels[i]>0 else None for i in range(1,6)]
            self.channels=[channels[i] if i is not None else 64 for i in self.indices]
        if any(i is None for i in self.indices[1:]):raise ValueError('Backbone lacks required multiscale stages')
        self.detail=nn.Sequential(nn.Conv2d(1,64,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU()) if self.indices[0] is None else None
        self.register_buffer('input_mean',torch.tensor(config['normalization']['mean']).reshape(1,3,1,1))
        self.register_buffer('input_std',torch.tensor(config['normalization']['std']).reshape(1,3,1,1))
        self.layout_nhwc=self.timm and 'swin_' in name
    def forward(self,x):
        raw=self.raw((x.repeat(1,3,1,1)-self.input_mean)/self.input_std);result=[]
        for level,index in enumerate(self.indices):
            z=self.detail(x) if index is None else raw[index]
            if self.layout_nhwc and index is not None:z=z.permute(0,3,1,2).contiguous()
            result.append(resize(z,(x.shape[-2]//2**(level+1),x.shape[-1]//2**(level+1))))
        return result

def encoder(name,pretrained,config):
    if name in ('resnet34','resnet50','resnet101'):
        return GrayEncoder(name,pretrained,**config['normalization'])
    if name=='rsu':return RSUBackbone(**config['normalization'])
    return Encoder(name,pretrained,config)

class Common(nn.Module):
    def __init__(self,config,pretrained):
        super().__init__();m=config['model'];self.target_size=config['train']['target'];self.m=m
        self.encoder=encoder(m['backbone'],pretrained,config)
        self.hybrid=None
        if m['hybrid_backbone']!='none':
            self.hybrid=encoder(m['hybrid_backbone'],pretrained,config)
            self.fusions=nn.ModuleList([nn.Conv2d(a+b,a,1) for a,b in zip(self.encoder.channels,self.hybrid.channels)])
        self.decoder=UNetDecoder(self.encoder.channels)
        # Additional heads never change base encoder/decoder RNG allocation.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config['train']['seed']+5101)
            if m['attention']!='scse':
                for block in list(self.decoder.blocks)+[self.decoder.final]:block.attention=attention(block.block[0].out_channels,m['attention'])
            if m['upsampling']=='pixelshuffle':self.decoder.upsampler=PixelShuffleUpsample()
            if m['skip_fusion']!='none':self.decoder.frequency_fusion=OffsetFusion() if m['skip_fusion']=='freq_offset' else FrequencyFusionNoOffset()
            kind=m['context'];self.context=CompressedContext(self.encoder.channels[-1],kind) if kind in ('aspp','lstm','cmaa') else None
            self.skip=WeightedSkip(self.encoder.channels) if kind=='weighted_skip' else AdjacentCross(self.encoder.channels[1:]) if kind=='cross_attention' else None
            self.detail=Snake(32) if kind=='snake' else FrequencyDetail(32) if kind=='frequency' else None
            self.ocr=OCR(32) if kind=='ocr' else None
            self.deep_heads=nn.ModuleList([nn.Conv2d(c,4,1) for c in (64,64,32)]) if m['deep_supervision'] else nn.ModuleList()
    def forward(self,x,return_features=False):
        if self.training and x.shape[-2:]!=(2*self.target_size,2*self.target_size):raise ValueError('Unexpected training context')
        features=self.encoder(x)
        if self.hybrid:features=[f(torch.cat([a,b],1)) for f,a,b in zip(self.fusions,features,self.hybrid(x))]
        if self.context:features[-1]=self.context(features[-1])
        if self.skip:
            features=self.skip(features) if self.m['context']=='weighted_skip' else [features[0]]+self.skip(features[1:])
        logits,stages=self.decoder(features,x.shape[-2:]);aux={}
        if self.detail:stages[-1]=self.detail(stages[-1]);logits=self.decoder.head(stages[-1])
        if self.ocr:
            stages[-1],coarse=self.ocr(stages[-1]);logits=self.decoder.head(stages[-1])
            if self.training:aux['ocr']=center_half(coarse)
        if self.training:
            logits=center_half(logits)
            if self.deep_heads:aux['deep']=[resize(center_half(head(feature)),logits.shape[-2:]) for head,feature in zip(self.deep_heads,stages)]
        return (logits,aux) if return_features else logits

class SmallU2(U2NETP):
    def __init__(self):
        super().__init__(3,4);self.attentions=nn.ModuleList([nn.Identity() for _ in range(5)])
    forward=FullU2Net.forward

class SegNet(nn.Module):
    def __init__(self):
        super().__init__();widths=[64,128,256,512,512];counts=[2,2,3,3,3]
        def block(cin,cout,count):
            layers=[]
            for _ in range(count):layers += [nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU()];cin=cout
            return nn.Sequential(*layers)
        self.enc=nn.ModuleList([block(a,b,n) for a,b,n in zip([3]+widths[:-1],widths,counts)])
        self.dec=nn.ModuleList([block(a,b,n) for a,b,n in zip(widths[::-1],[512,256,128,64,64],counts[::-1])]);self.head=nn.Conv2d(64,4,1)
    def forward(self,x):
        indices=[];sizes=[]
        for block in self.enc:
            x=block(x);sizes.append(x.size());x,idx=F.max_pool2d(x,2,return_indices=True);indices.append(idx)
        for block,idx,size in zip(self.dec,indices[::-1],sizes[::-1]):x=block(F.max_unpool2d(x,idx,2,output_size=size))
        return self.head(x)

class Native(nn.Module):
    def __init__(self,config,pretrained):
        super().__init__();m=config['model'];self.kind=m['architecture'];self.target_size=config['train']['target'];self.deep=m['deep_supervision']
        if self.kind=='u2net':self.network=FullU2Net(SCSE,False,config['train']['seed'])
        elif self.kind=='u2netp':self.network=SmallU2()
        elif self.kind=='segnet':self.network=SegNet()
        elif self.kind=='fcn':self.network=tv.segmentation.fcn_resnet101(weights=None,weights_backbone=tv.ResNet101_Weights.DEFAULT if pretrained else None,num_classes=4,aux_loss=False)
        else:
            import segmentation_models_pytorch as smp
            kwargs={'encoder_name':m['backbone'],'encoder_weights':'imagenet' if pretrained else None,'in_channels':3,'classes':4,'activation':None}
            if self.kind in ('Unet','UnetPlusPlus'):kwargs['decoder_attention_type']='scse'
            self.network=getattr(smp,self.kind)(**kwargs)
        self.register_buffer('input_mean',torch.tensor(config['normalization']['mean']).reshape(1,3,1,1))
        self.register_buffer('input_std',torch.tensor(config['normalization']['std']).reshape(1,3,1,1))
    def forward(self,x,return_features=False):
        if self.training and x.shape[-2:]!=(2*self.target_size,2*self.target_size):raise ValueError('Unexpected training context')
        z=(x.repeat(1,3,1,1)-self.input_mean)/self.input_std;aux={};pred=self.network(z)
        if self.kind.startswith('u2net'):
            pred,sides=pred
            if self.training and self.deep:aux['deep']=[center_half(v) for v in sides]
        elif isinstance(pred,dict):pred=pred['out']
        pred=resize(pred,x.shape[-2:])
        if self.training:pred=center_half(pred)
        return (pred,aux) if return_features else pred

def SegmentationModel(config,pretrained=False):
    m=config['model']
    if m['architecture']=='baseline':
        model=Baseline(config,pretrained=pretrained and m['initialization']=='imagenet')
        if pretrained and m['initialization'] in ('micronet','imagenet-micronet'):transfer_micronet(model,m['initialization'])
        return model
    if m['architecture']=='common':return Common(config,pretrained)
    return Native(config,pretrained)
