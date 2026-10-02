"""Train-only single-stage contrastive adaptation of the existing encoder."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.transforms import InterpolationMode, RandomResizedCrop, ColorJitter
from torchvision.transforms import functional as TF

from mimu.data import _patch_origins, _raw_gray

ENCODER_PREFIXES = ('conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.')


class GrayEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        first = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                first.weight.copy_(base.conv1.weight.sum(1, keepdim=True))
        self.conv1, self.bn1, self.relu, self.maxpool = first, base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2 = base.layer1, base.layer2
        self.layer3, self.layer4 = base.layer3, base.layer4
        self.register_buffer('input_mean', torch.tensor(.674325).reshape(1, 1, 1, 1))
        self.register_buffer('input_std', torch.tensor(.229455).reshape(1, 1, 1, 1))

    def forward(self, x):
        x = (x-self.input_mean)/self.input_std
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        return x.mean((-2, -1))


class ContrastiveEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = GrayEncoder(pretrained)
        self.projection = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                                        nn.ReLU(), nn.Linear(512, 128))

    def forward(self, x):
        return self.projection(self.encoder(x))


def nt_xent(z1, z2, temperature=.07):
    """Symmetric in-batch contrastive loss; each view's own logit is masked."""
    if z1.shape != z2.shape or z1.ndim != 2 or len(z1) < 2:
        raise ValueError('Two matching batches of at least two embeddings are required')
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    n = len(z1)
    z = F.normalize(torch.cat((z1, z2)), dim=1)
    logits = z @ z.T / temperature
    logits = logits.masked_fill(torch.eye(2*n, dtype=torch.bool, device=z.device), float('-inf'))
    labels = (torch.arange(2*n, device=z.device)+n) % (2*n)
    return F.cross_entropy(logits, labels)


class TrainViews:
    """Read Train images only, no masks; use distinct source images in each batch."""
    def __init__(self, root, patch=224):
        self.paths = sorted((Path(root)/'train/images').glob('*.png'))
        if not self.paths:
            raise ValueError('No Train images')
        self.images = [_raw_gray(p) for p in self.paths]
        if any(min(im.shape) < patch for im in self.images):
            raise ValueError('Train image smaller than crop')
        self.origins = [_patch_origins(*im.shape, patch, patch//2) for im in self.images]
        self.patch = patch
        self.crop = RandomResizedCrop(patch, scale=(.5, 1.), ratio=(.9, 1.1),
                                      interpolation=InterpolationMode.BILINEAR, antialias=True)
        self.jitter = ColorJitter(brightness=.1, contrast=.1)

    def view(self, anchor):
        image = self.crop(anchor)
        image = TF.rotate(image, 90*int(torch.randint(4, ())))
        if bool(torch.randint(2, ())):
            image = TF.hflip(image)
        return TF.to_tensor(self.jitter(image))

    def batch(self, size):
        if not 2 <= size <= len(self.paths):
            raise ValueError('Batch must contain 2..N distinct source images')
        sources = torch.randperm(len(self.paths))[:size]
        first, second = [], []
        for source in sources.tolist():
            origins = self.origins[source]
            y, x = origins[int(torch.randint(len(origins), ()))]
            anchor = TF.to_pil_image(self.images[source][y:y+self.patch, x:x+self.patch])
            first.append(self.view(anchor))
            second.append(self.view(anchor))
        return torch.stack(first), torch.stack(second), sources


def transfer_encoder(model, state):
    """Strict encoder-only replacement, including BN state; decoder is untouched."""
    expected = {k for k in model.state_dict() if k.startswith(ENCODER_PREFIXES)}
    actual = {k for k in state if k.startswith(ENCODER_PREFIXES)}
    if actual != expected:
        raise ValueError(f'Encoder keys differ: missing={expected-actual}, extra={actual-expected}')
    if any(not k.startswith(ENCODER_PREFIXES) and k not in ('input_mean', 'input_std') for k in state):
        raise ValueError('Unexpected non-encoder state')
    merged = model.state_dict()
    for k, tensor in state.items():
        if merged[k].shape != tensor.shape:
            raise ValueError(f'Encoder shape mismatch: {k}')
        merged[k] = tensor
    model.load_state_dict(merged, strict=True)
