import torch
from torch import nn
from torch.nn import functional as F


class HybridLoss(nn.Module):
    """Categorical focal or cross-entropy combined with multiclass soft Dice.

    alpha=.25, gamma=2 follow the focal library defaults in the author code.
    Dice uses equal class weights; MetalDAM-specific weights do not transfer.
    Focal reduction averages all N,C,H,W entries as in the author's library.
    """
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=2.0,
                 class_weights=(1.0, 1.0, 1.0, 1.0),
                 pixel_loss='focal', pixel_weight=1.0):
        super().__init__()
        if pixel_loss not in ('focal', 'ce'):
            raise ValueError("pixel_loss must be 'focal' or 'ce'")
        self.alpha, self.gamma = alpha, gamma
        self.dice_weight, self.pixel_loss, self.pixel_weight = dice_weight, pixel_loss, pixel_weight
        if len(class_weights) != 4 or any(v <= 0 for v in class_weights):
            raise ValueError('class_weights must contain four positive values')
        weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer('class_weights', weights / weights.mean())

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        if self.pixel_loss == 'ce':
            pixel = F.cross_entropy(logits, target)
        else:
            logpt = logp.gather(1, target[:, None]).squeeze(1)
            pixel = (-self.alpha * (1 - logpt.exp()).pow(self.gamma) * logpt).mean() / logits.shape[1]
        y = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).to(p.dtype)
        dims = (0, 2, 3)
        dice = (2 * (p * y).sum(dims) + 1e-5) / (p.sum(dims) + y.sum(dims) + 1e-5)
        weighted_dice_loss = ((1 - dice) * self.class_weights).sum() / self.class_weights.sum()
        return self.pixel_weight * pixel + self.dice_weight * weighted_dice_loss
