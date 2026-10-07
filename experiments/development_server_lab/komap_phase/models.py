import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

def resize(x, reference):
    size = reference.shape[-2:] if isinstance(reference, torch.Tensor) else reference
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class SCSE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, max(1, channels // 16), 1),
                                     nn.ReLU(), nn.Conv2d(max(1, channels // 16), channels, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel(x) + x * self.spatial(x)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, scse=True):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                                   nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())
        self.attention = SCSE(cout) if scse else nn.Identity()

    def forward(self, x):
        return self.attention(self.block(x))


class GrayEncoder(nn.Module):
    def __init__(self, backbone='resnet101', pretrained=False,
                 mean=(.485, .456, .406), std=(.229, .224, .225), output_stride=32):
        super().__init__()
        weights = {'resnet34': models.ResNet34_Weights, 'resnet50': models.ResNet50_Weights,
                   'resnet101': models.ResNet101_Weights}
        if backbone not in weights:
            raise ValueError('Unsupported ResNet backbone')
        options = {'replace_stride_with_dilation': [False, False, True]} if output_stride==16 else {}
        if backbone=='resnet34' and output_stride!=32:
            raise ValueError('ResNet34 debug supports OS32 only')
        base = getattr(models, backbone)(weights=weights[backbone].DEFAULT if pretrained else None, **options)
        # R022 repeats grayscale into RGB and keeps the original V2 weights.
        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.channels = [64, 64, 128, 256, 512] if backbone == 'resnet34' else [64, 256, 512, 1024, 2048]
        self.register_buffer('input_mean', torch.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor(std).reshape(1, 3, 1, 1))

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[1] != 3:
            raise ValueError('Expected grayscale (1 channel) or repeated RGB input')
        stem = self.relu(self.bn1(self.conv1((x - self.input_mean) / self.input_std)))
        e1 = self.layer1(self.maxpool(stem))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return [stem, e1, e2, e3, e4]


class ConvNeXtGrayEncoder(nn.Module):
    """New ConvNeXt-Tiny candidate, not an exact reconstruction of historical R024.

    Stages at OS4/8/16/32 plus a learned grayscale OS2 skip. ImageNet V1 weights.
    """
    def __init__(self, pretrained=False, mean=(.485,.456,.406), std=(.229,.224,.225)):
        super().__init__()
        base=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features=base.features
        self.detail=nn.Sequential(nn.Conv2d(1,64,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),
                                  nn.Conv2d(64,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU())
        self.channels=[64,96,192,384,768]
        self.register_buffer('input_mean',torch.tensor(mean).reshape(1,3,1,1))
        self.register_buffer('input_std',torch.tensor(std).reshape(1,3,1,1))

    def forward(self,x):
        if x.shape[1]!=1: raise ValueError('ConvNeXt candidate expects grayscale input')
        stages=[self.detail(x)]
        z=(x.repeat(1,3,1,1)-self.input_mean)/self.input_std
        for i,layer in enumerate(self.features):
            z=layer(z)
            if i in (1,3,5,7): stages.append(z)
        return stages


class BoundaryEnhancement(nn.Module):
    """Derived BEM: feature erosion difference, dilations1/2/3, gated residual.

    Zero output projection makes the initial segmentation exactly equal to B.
    This is not an implementation of the paper's full BE-UNet/RUCS framework.
    """
    def __init__(self, channels=64):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                                    groups=channels, bias=False),
                          nn.Conv2d(channels, channels, 1, bias=False), nn.ReLU())
            for d in (1, 2, 3)])
        self.gate = nn.Conv2d(channels, 1, 1)
        self.fuse = nn.Conv2d(channels*3, channels, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x):
        eroded = -F.max_pool2d(-F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
        edge = F.relu(x-eroded)
        context = self.fuse(torch.cat([branch(edge) for branch in self.branches], 1))
        return x + .1*self.gate(edge).sigmoid()*context


class FeatureRRCU(nn.Module):
    """Block replacement: compression, shared Conv-BN recurrence2, residual, scSE."""
    def __init__(self, cin, cout, attention):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(cin,cout,1,bias=False),nn.BatchNorm2d(cout),nn.ReLU())
        self.recurrent_conv = nn.Conv2d(cout,cout,3,padding=1,bias=False)
        self.recurrent_norm = nn.BatchNorm2d(cout)
        self.attention = attention

    def forward(self,x):
        r = self.project(x)
        h = torch.zeros_like(r)
        for _ in range(2):
            h = F.relu(self.recurrent_norm(self.recurrent_conv(r+h)))
        return self.attention(r+h)


class GateRRCU(nn.Module):
    """Zero-initialized residual logit branch; channel mode uses 1x1 after GAP.

    Spatial mode uses hidden16, shared3x3/GroupNorm4. Channel mode uses hidden4,
    shared1x1 without normalization at the pooled1x1 bottleneck. Both recur2.
    """
    def __init__(self, channels, mode):
        super().__init__()
        self.mode = mode
        hidden = 16 if mode=='spatial_gate' else max(1,channels//16)
        kernel = 3 if mode=='spatial_gate' else 1
        self.project = nn.Conv2d(channels,hidden,1)
        self.recurrent_conv = nn.Conv2d(hidden,hidden,kernel,padding=kernel//2)
        self.recurrent_norm = nn.GroupNorm(4,hidden) if mode=='spatial_gate' else nn.Identity()
        self.output = nn.Conv2d(hidden,1 if mode=='spatial_gate' else channels,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,x):
        if self.mode=='channel_gate':
            x = x.mean((-2,-1),keepdim=True)
        r = F.relu(self.project(x))
        h = torch.zeros_like(r)
        for _ in range(2):
            h = F.relu(self.recurrent_norm(self.recurrent_conv(r+h)))
        return self.output(r+h)


class RecurrentSCSE(nn.Module):
    """Keep both original gates and add recurrence to only one pre-sigmoid logit."""
    def __init__(self, original, channels, mode):
        super().__init__()
        self.channel = original.channel
        self.spatial = original.spatial
        self.mode = mode
        self.recurrent = GateRRCU(channels,mode)

    def forward(self,x):
        if self.mode=='spatial_gate':
            c = self.channel(x)
            s = (self.spatial[0](x)+.1*self.recurrent(x)).sigmoid()
        else:
            z = x
            for layer in list(self.channel.children())[:-1]:
                z = layer(z)
            c = (z+.1*self.recurrent(x)).sigmoid()
            s = self.spatial(x)
        return x*c+x*s


class UNetDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        incoming, widths = [channels[-1], 256, 128, 64], [256, 128, 64, 64]
        skips = list(reversed(channels[:-1]))
        self.blocks = nn.ModuleList([ConvBlock(cin+skip, cout) for cin, skip, cout in zip(incoming, skips, widths)])
        self.upsampler = nn.Identity()
        self.final = ConvBlock(64, 32)
        self.head = nn.Conv2d(32, 4, 1)

    def forward(self, features, size, bem=None):
        x, stages = features[-1], []
        for i, (skip, block) in enumerate(zip(reversed(features[:-1]), self.blocks)):
            x = block(torch.cat([resize(x, skip), skip], 1))
            if i == 3 and bem is not None:
                x = bem(x)
            if i in (2, 3):
                stages.append(x)
        fine = self.final(resize(x, size))
        stages.append(fine)
        return self.head(fine), stages


def center_half(x):
    h, w = x.shape[-2:]
    if h % 4 or w % 4:
        raise ValueError('Center feature crop requires dimensions divisible by four')
    return x[..., h//4:3*h//4, w//4:3*w//4]


class SegmentationModel(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.target_size = config['train']['target']
        m=config['model']
        if m['backbone']=='convnext_tiny':
            self.encoder=ConvNeXtGrayEncoder(pretrained,**config['normalization'])
        else:
            self.encoder=GrayEncoder(m['backbone'],pretrained,output_stride=m['output_stride'],**config['normalization'])
        self.decoder = UNetDecoder(self.encoder.channels)
        placement = config['model']['rrcu']
        if placement!='none':
            original = self.decoder.blocks[3]
            cin,cout = original.block[0].in_channels,original.block[0].out_channels
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config['train']['seed']+2301)
                if placement=='feature':
                    self.decoder.blocks[3] = FeatureRRCU(cin,cout,original.attention)
                else:
                    original.attention = RecurrentSCSE(original.attention,cout,placement)
        # Neither add-on changes baseline weight allocation nor global sampling RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config['train']['seed']+2101)
            self.bem = BoundaryEnhancement() if config['model']['bem'] else None
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config['train']['seed']+2102)
            width = config['aux']['embedding_width']
            self.projections = nn.ModuleList([
                nn.Sequential(nn.Conv2d(cin, width, 1), nn.ReLU(), nn.Conv2d(width, width, 1))
                for cin in (64, 64, 32)]) if config['model']['contrast'] else nn.ModuleList()

    def forward(self, x, return_features=False):
        if self.training and x.shape[-2:] != (2*self.target_size, 2*self.target_size):
            raise ValueError('Training expects configured context size')
        logits, stages = self.decoder(self.encoder(x), x.shape[-2:], self.bem)
        if self.training:
            logits = center_half(logits)
        if return_features:
            if not self.training:
                raise ValueError('Auxiliary features are training-only')
            embeddings = [head(center_half(feature)) for head, feature in zip(self.projections, stages)]
            return logits, embeddings
        return logits
