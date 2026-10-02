"""PyTorch port of the connected graph in the MIMU-Net authors' code.

Reference: sammajum706/Feature-Pyramid-Recurrent-Residual-U-Net-for-
Metallographic-Image-Segmentation, commit b2d21ef39cf6b513c8f3eb6d2df2dd76d59924ac.
See README.md for discrepancies between the paper and the released code.
"""
import torch
from torch import nn
from torch.nn import functional as F


def batch_norm(channels):
    # Keras defaults: epsilon=.001, moving-average momentum=.99.
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)


class ConvBNReLU(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Conv2d(channels, channels, 3, padding=1, bias=True),
                         batch_norm(channels), nn.ReLU())


class AuthorRecurrentUnit(nn.Module):
    """Preserve the authors' actual graph, including their x+x operation.

    Their loop creates two outputs from x+x but never feeds the first output
    back into x. Keras prunes that disconnected branch. The surviving unit is
    initial_conv(x), then final_conv(2 * initial_conv(x)); no shared weights.
    Replacing this with a conventional recurrent unit would change the model.
    """
    def __init__(self, channels):
        super().__init__()
        self.initial = ConvBNReLU(channels)
        self.final = ConvBNReLU(channels)

    def forward(self, x):
        x = self.initial(x)
        return self.final(x + x)


class RRCNNBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, channels, 1)
        self.units = nn.Sequential(AuthorRecurrentUnit(channels),
                                   AuthorRecurrentUnit(channels))

    def forward(self, x):
        x = self.projection(x)
        return x + self.units(x)


class AuthorSCSE(nn.Module):
    """Author implementation: C-channel spatial gate, summed cSE and sSE."""
    def __init__(self, channels):
        super().__init__()
        self.channel = nn.Sequential(nn.Linear(channels, channels // 2),
                                     nn.ReLU(), nn.Linear(channels // 2, channels),
                                     nn.Sigmoid())
        self.spatial = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        c = self.channel(x.mean(dim=(2, 3)))[:, :, None, None]
        s = self.spatial(x).sigmoid()
        return x * c + x * s


class MIMUNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        pairs = [(1, 14), (14, 28), (28, 56), (56, 112), (112, 224),
                 (224, 224), (224, 112), (224, 112), (112, 112), (112, 56),
                 (112, 56), (56, 56), (56, 28), (56, 28), (28, 28),
                 (28, 14), (210, 56), (56, 14)]
        self.blocks = nn.ModuleList([RRCNNBlock(a, b) for a, b in pairs])
        self.scse = nn.ModuleList([AuthorSCSE(c) for c in
                                   (14, 28, 56, 112, 224, 224, 112)])
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(14, num_classes, 1)
        self.head_bn = batch_norm(num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for se in self.scse:
            nn.init.kaiming_normal_(se.spatial.weight, mode='fan_in', nonlinearity='relu')

    @staticmethod
    def up(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or any(v % 16 for v in x.shape[-2:]):
            raise ValueError('Expected N,1,H,W with H and W divisible by 16')
        output_size = x.shape[-2:]
        skips = []
        for i in range(4):
            x = self.scse[i](self.blocks[i](x))
            skips.append(x)
            x = self.pool(x)
        for i in range(4, 7):
            x = self.scse[i](self.blocks[i](x))
        pyramid = [x]  # csse7: 112 channels at 1/16 resolution
        for start, skip in zip((7, 10, 13), reversed(skips[1:])):
            x = torch.cat([self.up(x, skip.shape[-2:]), skip], dim=1)
            for i in range(start, start + 3):
                x = self.blocks[i](x)
            pyramid.append(x)
        # Published order: conv16, conv13, conv10, csse7 -> 14+28+56+112.
        x = torch.cat([self.up(p, output_size) for p in reversed(pyramid)], dim=1)
        x = self.blocks[17](self.blocks[16](x))
        # Return logits; softmax is applied by loss/inference, never twice.
        return self.head_bn(self.head(x))
