"""Scoped paper-inspired ablations; not full paper reproductions.

Pixel contrast: Wang et al., ICCV 2021, supervised cross-image pixel features.
Snake branch: Qi et al., ICCV 2023, perpendicular cumulative offsets.
Boundary head: shape-supervision idea from Gated-SCNN, ICCV 2019 (no gates).
clDice: Shit et al., CVPR 2021, soft morphological skeleton overlap.
"""
import torch
from torch import nn
from torch.nn import functional as F

from mimu.resnet34_unet import ResNet34UNet


def bilinear_sample(feature, sx, sy):
    """Border-clamped pixel coordinates; gather form supports MPS backward."""
    n, c, h, w = feature.shape
    sx, sy = sx.clamp(0, w-1), sy.clamp(0, h-1)
    x0, y0 = sx.floor().long(), sy.floor().long()
    x1, y1 = (x0+1).clamp(max=w-1), (y0+1).clamp(max=h-1)
    wx, wy = (sx-x0.to(sx.dtype))[:, None], (sy-y0.to(sy.dtype))[:, None]
    flat = feature.flatten(2)
    def at(iy, ix):
        indices = (iy*w+ix).flatten(1)[:, None].expand(-1, c, -1)
        return flat.gather(2, indices).reshape(n, c, h, w)
    return ((1-wx)*(1-wy)*at(y0,x0) + wx*(1-wy)*at(y0,x1)
            + (1-wx)*wy*at(y1,x0) + wx*wy*at(y1,x1))


class SnakeBranch(nn.Module):
    """5-tap horizontal/vertical snakes at 1/2 input resolution.

    Center is fixed; each neighbor accumulates bounded perpendicular offsets.
    Sampling is bilinear, with border padding. Ordinary decoder path is retained.
    """
    def __init__(self, channels=64, width=16, radius=2):
        super().__init__()
        self.radius = radius
        self.reduce = nn.Conv2d(channels, width, 1)
        self.offset = nn.Conv2d(channels, 4 * radius, 3, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.project = nn.Sequential(
            nn.Conv2d(width * (2 * radius + 1) * 2, channels, 1, bias=False),
            nn.GroupNorm(8, channels), nn.ReLU())
        # Initially preserves the ordinary feature, then learns the snake contribution.
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        n, _, h, w = x.shape
        feature = self.reduce(x)
        delta = self.offset(x).tanh()
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                                torch.arange(w, device=x.device, dtype=x.dtype), indexing='ij')
        samples = []
        for axis in range(2):
            curves = {0: torch.zeros_like(delta[:, 0])}
            for sign_index, sign in enumerate((-1, 1)):
                cumulative = torch.zeros_like(delta[:, 0])
                for distance in range(1, self.radius + 1):
                    channel = axis * 2 * self.radius + sign_index * self.radius + distance - 1
                    cumulative = cumulative + delta[:, channel]
                    curves[sign * distance] = cumulative
            for step in range(-self.radius, self.radius + 1):
                if axis == 0:
                    sx, sy = xx[None] + step, yy[None] + curves[step]
                else:
                    sx, sy = xx[None] + curves[step], yy[None] + step
                sx = sx.expand(n, h, w)
                sy = sy.expand(n, h, w)
                samples.append(bilinear_sample(feature, sx, sy))
        return x + self.gain.tanh() * self.project(torch.cat(samples, dim=1))


class SnakeDecoder(nn.Module):
    def __init__(self, ordinary):
        super().__init__()
        self.ordinary = ordinary
        self.snake = SnakeBranch()

    def forward(self, x):
        return self.snake(self.ordinary(x))


class ResearchUNet(ResNet34UNet):
    def __init__(self, method, pretrained=True):
        super().__init__(pretrained=pretrained, use_scse=True)
        self.method = method
        # Keep common model initialization and data RNG identical across variants.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1701)
            if method == 'contrast':
                self.projection = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
            elif method == 'snake':
                self.decoder1 = SnakeDecoder(self.decoder1)
            elif method == 'boundary':
                self.boundary_head = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1),
                                                   nn.ReLU(), nn.Conv2d(16, 4, 1))
            elif method not in ('cldice', 'baseline'):
                raise ValueError(method)

    def forward(self, x, auxiliary=False):
        logits, features = super().forward(x, return_features=True)
        if not auxiliary:
            return logits
        return logits, features


def class_boundaries(target):
    masks = F.one_hot(target, 4).permute(0, 3, 1, 2).float()
    dilated = F.max_pool2d(masks, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-masks, 3, stride=1, padding=1)
    return dilated - eroded


def boundary_loss(model, features, target):
    edges = class_boundaries(target)
    logits = model.boundary_head(features)
    fraction = edges.mean(dim=(0, 2, 3))
    pos_weight = ((1-fraction) / fraction.clamp_min(1e-6)).clamp(1, 10)[None, :, None, None]
    return F.binary_cross_entropy_with_logits(logits, edges, pos_weight=pos_weight)


def pixel_contrast(model, features, logits, target, sources, generator, per_class=32, temperature=0.1):
    """Sample hard/easy class-balanced interior pixels; cross-source positives.

    Standard supervised contrastive denominator, no memory bank. Independent CPU
    sampling generator prevents auxiliary sampling from changing crop selection.
    """
    truth = target.detach().cpu()
    prediction = logits.detach().argmax(1).cpu()
    interior = (class_boundaries(target).sum(1) == 0).detach().cpu()
    vectors, classes, origins = [], [], []
    for image in range(len(truth)):
        flat = features[image].flatten(1).transpose(0, 1)
        for cls in range(4):
            mask = (truth[image] == cls) & interior[image]
            hard = (mask & (prediction[image] != cls)).flatten().nonzero().flatten()
            easy = (mask & (prediction[image] == cls)).flatten().nonzero().flatten()
            count = min(per_class, len(hard) + len(easy))
            if count == 0:
                continue
            nh = min(len(hard), count // 2)
            ne = min(len(easy), count - nh)
            nh = min(len(hard), count - ne)
            idx = torch.cat((hard[torch.randperm(len(hard), generator=generator)[:nh]],
                             easy[torch.randperm(len(easy), generator=generator)[:ne]]))
            vectors.append(flat[idx.to(features.device)])
            classes.extend([cls] * len(idx))
            origins.extend([int(sources[image])] * len(idx))
    if not vectors:
        return features.sum() * 0
    embedding = F.normalize(model.projection(torch.cat(vectors)), dim=1)
    labels = torch.tensor(classes, device=embedding.device)
    source = torch.tensor(origins, device=embedding.device)
    similarity = embedding @ embedding.T / temperature
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=embedding.device)
    positives = (labels[:, None] == labels[None]) & (source[:, None] != source[None])
    positive_count = positives.sum(1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return embedding.sum() * 0
    log_prob = similarity - similarity.masked_fill(diagonal, -1e4).logsumexp(1, keepdim=True)
    return -((log_prob * positives).sum(1) / positive_count.clamp_min(1))[valid].mean()


def soft_erode(mask):
    return torch.minimum(-F.max_pool2d(-mask, (3, 1), 1, (1, 0)),
                         -F.max_pool2d(-mask, (1, 3), 1, (0, 1)))


def soft_skeleton(mask, iterations=8):
    opened = F.max_pool2d(soft_erode(mask), 3, 1, 1)
    skeleton = F.relu(mask - opened)
    for _ in range(iterations):
        mask = soft_erode(mask)
        opened = F.max_pool2d(soft_erode(mask), 3, 1, 1)
        delta = F.relu(mask - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cldice_loss(logits, target):
    probability = logits.softmax(1)[:, 2:3]
    truth = (target[:, None] == 2).float()
    sk_pred = soft_skeleton(probability)
    sk_true = soft_skeleton(truth)
    dims = (1, 2, 3)
    precision = ((sk_pred * truth).sum(dims) + 1e-6) / (sk_pred.sum(dims) + 1e-6)
    recall = ((sk_true * probability).sum(dims) + 1e-6) / (sk_true.sum(dims) + 1e-6)
    present = truth.sum(dims) > 0
    if not bool(present.any()):
        return logits.sum() * 0
    score = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    return (1-score[present]).mean()
