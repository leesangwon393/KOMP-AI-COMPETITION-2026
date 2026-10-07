"""Paper-inspired hybrids, preserving the exact baseline module allocation."""
import hashlib
import math
import torch
from torch import nn
from torchvision import models
from .models import SegmentationModel as Baseline, resize, center_half


def weights_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def projection(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.GroupNorm(32, cout), nn.GELU())


class ResidualFusion(nn.Module):
    def __init__(self, cnn, swin, width, alpha):
        super().__init__()
        self.cnn = projection(cnn, width)
        self.swin = projection(swin, width // 2)
        self.mix = projection(width + width // 2, width)
        self.output = nn.Conv2d(width, cnn, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.alpha = alpha

    def forward(self, cnn, swin):
        if cnn.shape[-2:] != swin.shape[-2:]:
            raise ValueError('CNN/Swin spatial stages do not match')
        fused = self.mix(torch.cat([self.cnn(cnn), self.swin(swin)], 1))
        return cnn + self.alpha * self.output(fused)


class SwinFeatures(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        base = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features, self.norm = base.features, base.norm

    def forward(self, x):
        stages = []
        for index, block in enumerate(self.features):
            x = block(x)
            if index in (1, 3, 5, 7):
                stage = self.norm(x) if index == 7 else x
                stages.append(stage.permute(0, 3, 1, 2).contiguous())
        return stages


def positional_encoding(height, width, channels, device, dtype):
    if channels % 4:
        raise ValueError('2D sinusoidal channels must be divisible by4')
    frequency = torch.exp(-math.log(10000.) * torch.arange(channels // 4, device=device).float() / (channels // 4))
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    y, x = y.flatten()[:, None] * frequency, x.flatten()[:, None] * frequency
    return torch.cat([y.sin(), y.cos(), x.sin(), x.cos()], 1).to(dtype)[None]


class DecoderAttention(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(256, 4, 1024, dropout=0.,
            activation='gelu', batch_first=True, norm_first=True) for _ in range(2)])
        self.output = nn.Conv2d(256, 256, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.alpha = alpha

    def forward(self, x):
        n, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + positional_encoding(h, w, c, x.device, x.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        context = tokens.transpose(1, 2).reshape(n, c, h, w)
        return x + self.alpha * self.output(context)


class SegmentationModel(Baseline):
    def __init__(self, config, pretrained=False):
        super().__init__(config, pretrained)
        self.variant = config['transformer']['variant']
        self.freeze_epochs = config['transformer']['freeze_epochs']
        self.epoch = 1
        self.swin = None
        self.fusions = nn.ModuleDict()
        self.decoder_attention = None
        branch_seed = config['train']['seed'] + config['transformer']['branch_seed_offset']
        self.branch_cpu_rng = torch.Generator().manual_seed(branch_seed).get_state()
        self.branch_cuda_rng = None
        self.branch_seed = branch_seed
        self.initialization = {'cnn_weights': 'ResNet101_Weights.IMAGENET1K_V2' if pretrained else 'none',
                               'common_initial_sha256': weights_hash(self.state_dict())}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(branch_seed)
            alpha = config['transformer']['residual_scale']
            if self.variant in ('HT_BOT', 'HT_PYR'):
                self.swin = SwinFeatures(pretrained)
                for stage in ([4] if self.variant == 'HT_BOT' else [2, 3, 4]):
                    self.fusions[str(stage)] = ResidualFusion(
                        [64, 256, 512, 1024, 2048][stage], [96, 192, 384, 768][stage-1],
                        [128, 256, 512][stage-2], alpha)
                self.initialization.update(swin_weights='Swin_T_Weights.IMAGENET1K_V1' if pretrained else 'none',
                                           swin_initial_sha256=weights_hash(self.swin.state_dict()))
            elif self.variant == 'HT_DEC16':
                self.decoder_attention = DecoderAttention(alpha)
        self.set_epoch(1)

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.swin is not None:
            frozen = epoch <= self.freeze_epochs
            self.swin.requires_grad_(not frozen)
            self.swin.train(self.training and not frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.swin is not None and self.epoch <= self.freeze_epochs:
            self.swin.eval()
        return self

    def branch_state(self):
        return {'cpu': self.branch_cpu_rng.clone(), 'cuda': self.branch_cuda_rng,
                'epoch': self.epoch, 'swin_frozen': self.swin is not None and self.epoch <= self.freeze_epochs}

    def restore_branch(self, state, epoch):
        expected = self.swin is not None and epoch <= self.freeze_epochs
        if state['epoch'] != epoch or state['swin_frozen'] != expected:
            raise ValueError('Checkpoint freeze phase differs')
        self.branch_cpu_rng = state['cpu'].cpu()
        self.branch_cuda_rng = state['cuda'].cpu() if state['cuda'] is not None else None
        self.set_epoch(epoch)

    def swin_forward(self, x):
        # Swin stochastic depth must not consume the CPU RNG used by dataset sampling.
        devices = [x.device.index if x.device.index is not None else torch.cuda.current_device()] if x.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.branch_cpu_rng)
            if x.is_cuda:
                if self.branch_cuda_rng is None:
                    self.branch_cuda_rng = torch.Generator(device=x.device).manual_seed(self.branch_seed).get_state()
                torch.cuda.set_rng_state(self.branch_cuda_rng, x.device)
            rgb = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
            features = self.swin((rgb - self.encoder.input_mean) / self.encoder.input_std)
            self.branch_cpu_rng = torch.get_rng_state()
            if x.is_cuda:
                self.branch_cuda_rng = torch.cuda.get_rng_state(x.device)
        return features

    def forward(self, x, return_features=False):
        if self.training and x.shape[-2:] != (2*self.target_size, 2*self.target_size):
            raise ValueError('Training expects configured context size')
        features = self.encoder(x)
        if self.swin is not None:
            transformer = self.swin_forward(x)
            for stage, fuse in self.fusions.items():
                index = int(stage)
                features[index] = fuse(features[index], transformer[index-1])
        if self.decoder_attention is None:
            logits, _ = self.decoder(features, x.shape[-2:], self.bem)
        else:
            z = features[-1]
            for index, (skip, block) in enumerate(zip(reversed(features[:-1]), self.decoder.blocks)):
                z = block(torch.cat([resize(z, skip), skip], 1))
                if index == 0:
                    z = self.decoder_attention(z)
            logits = self.decoder.head(self.decoder.final(resize(z, x.shape[-2:])))
        if self.training:
            logits = center_half(logits)
        if return_features:
            if not self.training:
                raise ValueError('Training features requested in eval mode')
            return logits, []
        return logits


def make_optimizer(model, config):
    t = config['train']
    if model.swin is None:
        return torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    # Include frozen weights from the start, with a fixed order across resumes.
    groups = [dict(params=list(model.encoder.parameters()), name='cnn', lr=t['lr']),
              dict(params=list(model.decoder.parameters()), name='decoder', lr=t['lr']),
              dict(params=list(model.fusions.parameters()), name='fusion', lr=t['lr']),
              dict(params=list(model.swin.parameters()), name='swin', lr=config['transformer']['swin_lr'])]
    return torch.optim.AdamW(groups, weight_decay=t['weight_decay'])
