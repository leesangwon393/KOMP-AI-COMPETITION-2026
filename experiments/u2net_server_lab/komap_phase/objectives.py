"""Train-label-only auxiliary objectives; no validation/test pseudo labels."""
import torch
from torch import nn
from torch.nn import functional as F

from .loss import R022Loss


def contact_band(target, radius=2):
    """Pixels of classes1/2 around their direct 4-neighbor contact, no wraparound."""
    padded = F.pad(target, (1, 1, 1, 1), value=255)
    contact = torch.zeros_like(target, dtype=torch.bool)
    h, w = target.shape[-2:]
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        neighbor = padded[..., 1+dy:1+dy+h, 1+dx:1+dx+w]
        contact |= ((target == 1) & (neighbor == 2)) | ((target == 2) & (neighbor == 1))
    if radius:
        contact = F.max_pool2d(contact.float()[:, None], 2*radius+1, stride=1, padding=radius)[:, 0] > 0
    return contact & ((target == 1) | (target == 2))


def weighted_contact_ce(logits, target, multiplier, radius):
    ce = F.cross_entropy(logits, target, reduction='none')
    band = contact_band(target, radius)
    weight = 1+(multiplier-1)*band.to(ce.dtype)
    loss = ((ce*weight).sum((-2, -1))/weight.sum((-2, -1))).mean()
    return loss, band.float().mean()


def pure_labels(target, size, threshold):
    one_hot = F.one_hot(target, 4).permute(0, 3, 1, 2).float()
    fractions = F.adaptive_avg_pool2d(one_hot, size)
    fraction, labels = fractions.max(1)
    return labels, fraction >= threshold


def balanced_contrast(embedding, target, generator, samples_per_class=32, purity=.9, temperature=.1):
    labels, pure = pure_labels(target, embedding.shape[-2:], purity)
    flat_labels = labels.flatten().cpu()
    flat_pure = pure.flatten().cpu()
    picked, classes = [], []
    for cls in range(4):
        candidates = torch.where(flat_pure & (flat_labels == cls))[0]
        if len(candidates) < 2:
            continue
        chosen = candidates[torch.randperm(len(candidates), generator=generator)[:samples_per_class]]
        picked.append(chosen)
        classes.append(torch.full((len(chosen),), cls, dtype=torch.long))
    if len(picked) < 2:
        return embedding.sum()*0., {'points': 0, 'classes': len(picked)}
    indices = torch.cat(picked).to(embedding.device)
    y = torch.cat(classes).to(embedding.device)
    vectors = embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1])[indices]
    vectors = F.normalize(vectors, dim=1, eps=1e-8)
    similarity = vectors @ vectors.T / temperature
    diagonal = torch.eye(len(y), dtype=torch.bool, device=y.device)
    positives = (y[:, None] == y[None, :]) & ~diagonal
    # Finite diagonal filler avoids 0 * -inf in the positive reduction.
    denominator = torch.logsumexp(similarity.masked_fill(diagonal, -1e4), dim=1)
    log_probability = similarity-denominator[:, None]
    anchor_loss = -(log_probability*positives).sum(1)/positives.sum(1).clamp_min(1)
    # Equal class contribution even if fewer than the cap are available.
    loss = torch.stack([anchor_loss[y == cls].mean() for cls in torch.unique(y)]).mean()
    return loss, {'points': len(y), 'classes': len(picked)}


class PhaseLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base = R022Loss(**config['loss'])
        self.aux = config['aux']
        self.deep_supervision=config['model']['deep_supervision']
        self.generator = torch.Generator(device='cpu').manual_seed(config['train']['seed']+2103)

    def forward(self, logits, target, embeddings, epoch):
        base_loss = self.base(logits, target)
        contact_fraction = logits.new_zeros(())
        if self.aux['contact_weight'] != 1:
            weighted, contact_fraction = weighted_contact_ce(logits, target, self.aux['contact_weight'], self.aux['contact_radius'])
            base_loss = base_loss + .5*(weighted-F.cross_entropy(logits, target))
        side_loss=logits.new_zeros(())
        total=base_loss
        if self.deep_supervision:
            if len(embeddings)!=6: raise ValueError('Deep supervision requires6 aligned side logits')
            if any(side.shape!=logits.shape for side in embeddings): raise ValueError('Misaligned side logits')
            side_loss=torch.stack([self.base(side,target) for side in embeddings]).mean()
            weight=self.aux['deep_side_weight']
            total=(base_loss+len(embeddings)*weight*side_loss)/(1+len(embeddings)*weight)
        elif embeddings:
            raise ValueError('Unexpected auxiliary heads')
        return total, {'seg_loss':float(base_loss.detach()),'contrast_loss':0.,'contrast_lambda':0.,
                       'contrast_points':0.,'deep_supervision_loss':float(side_loss.detach()),
                       'contact_fraction':float(contact_fraction.detach())}
