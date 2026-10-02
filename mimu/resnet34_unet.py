"""ResNet34 U-Net with an optional scSE attention block in each decoder stage."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights, resnet34


class SCSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel(x) + x * self.spatial(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_scse):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.attention = SCSE(out_channels) if use_scse else nn.Identity()

    def forward(self, x):
        return self.attention(self.block(x))


class ResNet34UNet(nn.Module):
    """Grayscale ResNet34 U-Net; ImageNet weights are optional for ablations."""
    def __init__(self, num_classes=4, pretrained=True, use_scse=True):
        super().__init__()
        encoder = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        old = encoder.conv1
        first = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                # A grayscale input repeated into RGB produces the sum of RGB kernels.
                first.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        self.conv1, self.bn1, self.relu, self.maxpool = first, encoder.bn1, encoder.relu, encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.decoder4 = ConvBlock(512 + 256, 256, use_scse)
        self.decoder3 = ConvBlock(256 + 128, 128, use_scse)
        self.decoder2 = ConvBlock(128 + 64, 64, use_scse)
        self.decoder1 = ConvBlock(64 + 64, 64, use_scse)
        self.decoder0 = ConvBlock(64, 32, use_scse)
        self.head = nn.Conv2d(32, num_classes, 1)
        self.register_buffer('input_mean', torch.tensor(0.674325).view(1, 1, 1, 1))
        self.register_buffer('input_std', torch.tensor(0.229455).view(1, 1, 1, 1))
        self.pretrained = pretrained
        self.use_scse = use_scse

    def forward(self, x, return_features=False):
        input_size = x.shape[-2:]
        if self.pretrained:
            x = (x - self.input_mean) / self.input_std
        stem = self.relu(self.bn1(self.conv1(x)))
        e1 = self.layer1(self.maxpool(stem))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        x = self.layer4(e3)
        for skip, decoder in ((e3, self.decoder4), (e2, self.decoder3),
                              (e1, self.decoder2), (stem, self.decoder1)):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        features = self.decoder0(x)
        logits = self.head(features)
        return (logits, features) if return_features else logits
