"""Train-label-only auxiliary objectives; no validation/test pseudo labels."""
import torch
from torch import nn
from torch.nn import functional as F

from .loss import R022Loss
from .followup_modules import connectivity_loss


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
        self.generator = torch.Generator(device='cpu').manual_seed(config['train']['seed']+2103)

    def forward(self, logits, target, embeddings, epoch):
        extra = embeddings if isinstance(embeddings, dict) else {}
        embeddings = [] if isinstance(embeddings, dict) else embeddings
        base_loss = self.base(logits, target)
        contact_fraction = logits.new_zeros(())
        if self.aux['contact_weight'] != 1:
            weighted, contact_fraction = weighted_contact_ce(logits, target, self.aux['contact_weight'], self.aux['contact_radius'])
            base_loss = base_loss + self.base.ce_weight*(weighted-F.cross_entropy(logits, target))
        ramp = max(0., min(1., (epoch-self.aux['warmup_epochs'])/self.aux['ramp_epochs']))
        weight = self.aux['contrast_weight']*ramp
        local = logits.new_zeros(())
        points = 0
        if weight and embeddings:
            losses = []
            for embedding in embeddings:
                loss, stats = balanced_contrast(embedding, target, self.generator,
                    self.aux['samples_per_class'], self.aux['purity'], self.aux['temperature'])
                # Empty levels don't dilute usable levels.
                if stats['points']:
                    losses.append(loss)
                    points += stats['points']
            if losses:
                local = torch.stack(losses).mean()
        connection=logits.new_zeros(())
        if self.aux['connectivity_weight']:
            if len(embeddings)!=1: raise ValueError('Expected one connectivity auxiliary output')
            connection=connectivity_loss(embeddings[0],target)
        deep=logits.new_zeros(());consistency=logits.new_zeros(());ocr=logits.new_zeros(())
        if self.aux['deep_weight']:
            heads=extra.get('deep',[])
            if not heads:raise ValueError('Missing supervised deep heads')
            deep=torch.stack([self.base(head,target) for head in heads]).mean()
            if self.aux['consistency_weight']:
                teacher=logits.detach().softmax(1)
                consistency=torch.stack([(head.softmax(1)-teacher).square().mean() for head in heads]).mean()
        if self.aux['ocr_weight']:
            if 'ocr' not in extra:raise ValueError('Missing OCR region head')
            ocr=F.cross_entropy(extra['ocr'],target)
        total = base_loss + weight*local + self.aux['connectivity_weight']*connection
        total=total+self.aux['deep_weight']*deep+self.aux['consistency_weight']*consistency+self.aux['ocr_weight']*ocr
        return total, {'seg_loss': float(base_loss.detach()), 'contrast_loss': float(local.detach()),
                       'contrast_lambda': weight, 'contrast_points': points,
                       'contact_fraction': float(contact_fraction.detach()),'connectivity_loss':float(connection.detach()),
                       'deep_loss':float(deep.detach()),'consistency_loss':float(consistency.detach()),'ocr_loss':float(ocr.detach())}
