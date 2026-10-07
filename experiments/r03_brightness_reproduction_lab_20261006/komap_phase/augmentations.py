"""Paired augmentation ablations: preserve legacy sampling/geometry RNG consumption."""
import math

import torch
from torch.nn import functional as F

from .data import NativePatchDataset


class AugmentedNativePatchDataset(NativePatchDataset):
    def __init__(self, *args, augmentation, seed, **kwargs):
        self.augmentation = dict(augmentation)
        self.extra_generator = torch.Generator(device='cpu').manual_seed(seed+augmentation['extra_seed_offset'])
        super().__init__(*args, **kwargs)

    @staticmethod
    def blur(image, sigma):
        radius = max(1, math.ceil(3*sigma))
        axis = torch.arange(-radius,radius+1,dtype=image.dtype,device=image.device)
        kernel = torch.exp(-axis.square()/(2*sigma*sigma))
        kernel /= kernel.sum()
        x = F.pad(image[None],(radius,radius,0,0),mode='reflect')
        x = F.conv2d(x,kernel.view(1,1,1,-1))
        x = F.pad(x,(0,0,radius,radius),mode='reflect')
        return F.conv2d(x,kernel.view(1,1,-1,1))[0]

    def _photometric(self, image):
        if not self.photometric:
            return image
        a = self.augmentation
        # Draw exactly the same legacy random numbers even when a transform is off.
        # This keeps source/crop/D4/noise realizations paired across all conditions.
        if torch.rand(()) < .5:
            contrast = torch.empty(()).uniform_(.9,1.1)
            brightness = torch.empty(()).uniform_(-.08,.08)
            if a['contrast_amplitude'] != .10:
                contrast = 1+(contrast-1)*(a['contrast_amplitude']/.10)
            if a['brightness_amplitude'] != .08:
                brightness = brightness*(a['brightness_amplitude']/.08)
            if a['contrast_amplitude'] or a['brightness_amplitude']:
                mean = image.mean((-2,-1),keepdim=True)
                image = (image-mean)*contrast+mean+brightness
        if torch.rand(()) < .3:
            gamma = torch.empty(()).uniform_(.9,1.1)
            if a['gamma_amplitude'] != .10:
                gamma = 1+(gamma-1)*(a['gamma_amplitude']/.10)
            image = image.clamp(0,1).pow(gamma)
        if torch.rand(()) < .15:
            noise = torch.randn_like(image)
            image = image+noise*a['noise_sigma']
        image = image.clamp(0,1)
        # Additional transforms use a separately checkpointed CPU RNG.
        draws = torch.rand(4,generator=self.extra_generator)
        if float(draws[0]) < a['illumination_probability']:
            angle = float(draws[1])*2*math.pi
            dx,dy = math.cos(angle),math.sin(angle)
            x = torch.linspace(-1,1,image.shape[-1],dtype=image.dtype)[None,None,:]
            y = torch.linspace(-1,1,image.shape[-2],dtype=image.dtype)[None,:,None]
            field = (dx*x+dy*y)/(abs(dx)+abs(dy))
            image = (image+a['illumination_amplitude']*field).clamp(0,1)
        if float(draws[2]) < a['blur_probability']:
            lo,hi = a['blur_sigma']
            image = self.blur(image,lo+(hi-lo)*float(draws[3]))
        return image.clamp(0,1)
