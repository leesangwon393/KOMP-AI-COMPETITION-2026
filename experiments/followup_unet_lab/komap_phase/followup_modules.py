"""Pure PyTorch follow-up modules; derived designs, not full paper replicas."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def reassemble(feature, kernels, kernel_size, scale=1):
    """Reflect-padded adaptive filtering; low-res neighborhoods precede upsampling.

    Shift accumulation avoids the full B*C*K*K*H*W unfold allocation.
    Its values match group1 CARAFE's public PyTorch reference.
    """
    b, c, h, w = feature.shape
    if kernels.shape != (b, kernel_size**2, h*scale, w*scale):
        raise ValueError('Adaptive kernel shape differs from feature/scale')
    padded = F.pad(feature, (kernel_size//2,)*4, mode='reflect')
    result = feature.new_zeros(b, c, h*scale, w*scale)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            part = padded[..., dy:dy+h, dx:dx+w]
            if scale != 1:
                part = F.interpolate(part, scale_factor=scale, mode='nearest')
            result = result + part * kernels[:, dy*kernel_size+dx:dy*kernel_size+dx+1]
    return result


class FrequencyFusionNoOffset(nn.Module):
    """FreqFusion-inspired ALPF5/AHPF3 at the shallow OS4->OS2 skip.

    Two channel compressors, summed compressed features, Hamming-normalized
    dynamic kernels, adaptive low-res reassembly x2 and high-res residual.
    No offsets or iterative compressed-feature initialization. Hence a reduced
    FreqFusion-derived configuration, not the full published model.
    """
    def __init__(self, high_channels=64, low_channels=64, compressed=32):
        super().__init__()
        self.high = nn.Conv2d(high_channels, compressed, 1)
        self.low = nn.Conv2d(low_channels, compressed, 1)
        self.low_kernel = nn.Conv2d(compressed, 25, 3, padding=1)
        self.high_kernel = nn.Conv2d(compressed, 9, 3, padding=1)
        for layer in (self.low_kernel, self.high_kernel):
            nn.init.normal_(layer.weight, std=.001)
            nn.init.zeros_(layer.bias)
        for size in (3, 5):
            window = torch.hamming_window(size, periodic=False)
            self.register_buffer(f'window{size}', (window[:, None]*window[None]).reshape(1, size*size, 1, 1))

    def normalize(self, logits, size):
        kernels = logits.softmax(1)*getattr(self, f'window{size}')
        return kernels/kernels.sum(1, keepdim=True)

    def _forward(self, high, low):
        if high.shape[-2:] != (low.shape[-2]*2, low.shape[-1]*2):
            raise ValueError('Frequency fusion requires exactly x2 spatial ratio')
        shared = self.high(high)+F.interpolate(self.low(low), size=high.shape[-2:], mode='nearest')
        low_mask = self.normalize(self.low_kernel(shared), 5)
        high_mask = self.normalize(self.high_kernel(shared), 3)
        return 2*high-reassemble(high, high_mask, 3), reassemble(low, low_mask, 5, 2)

    def forward(self, high, low):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, high, low, use_reentrant=False)
        return self._forward(high, low)


class PixelShuffleUpsample(nn.Module):
    """Learned OS2->OS1 upsampling, preserving the existing final ConvBlock."""
    def __init__(self, channels=64):
        super().__init__()
        self.project = nn.Conv2d(channels, 4*channels, 1)
        # ICNR-style equal subpixel initialization prevents an initial checkerboard.
        with torch.no_grad():
            base = torch.empty(channels, channels, 1, 1)
            nn.init.kaiming_normal_(base)
            self.project.weight.copy_(base.repeat_interleave(4, 0))
            self.project.bias.zero_()

    def forward(self, x):
        return F.pixel_shuffle(self.project(x), 2)


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def connectivity_targets(target):
    """Channel order class*8+direction. Include class0; ignore outside center GT."""
    h, w = target.shape[-2:]
    padded = F.pad(target, (1, 1, 1, 1), value=255)
    one_hot = F.one_hot(target, 4).permute(0, 3, 1, 2).bool()
    targets, valid = [], []
    for dy, dx in DIRECTIONS:
        neighbor = padded[..., 1+dy:1+dy+h, 1+dx:1+dx+w]
        targets.append(one_hot & (neighbor[:, None] == target[:, None]))
        valid.append((neighbor != 255)[:, None].expand(-1, 4, -1, -1))
    return torch.stack(targets, 2).flatten(1, 2).float(), torch.stack(valid, 2).flatten(1, 2)


def connectivity_loss(logits, target):
    desired, valid = connectivity_targets(target)
    if logits.shape != desired.shape:
        raise ValueError('Connectivity logits must match four classes x eight directions')
    bce = F.binary_cross_entropy_with_logits(logits, desired, reduction='none')
    # Equal class/direction contribution and positive/negative contributions where
    # both exist. Absent classes still receive negative supervision.
    positive = valid & desired.bool()
    negative = valid & ~desired.bool()
    terms = []
    for mask in (positive, negative):
        count = mask.sum((0, 2, 3))
        terms.append(((bce*mask).sum((0, 2, 3))/count.clamp_min(1), count > 0))
    numerator = sum(value*present for value, present in terms)
    denominator = sum(present.int() for _, present in terms).clamp_min(1)
    return (numerator/denominator).mean()
