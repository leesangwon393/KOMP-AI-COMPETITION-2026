from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ['Primary Si', 'Al3Ni', 'Eutectic Si', 'Al']
PALETTE = np.array([[76, 178, 76], [25, 76, 153], [204, 204, 204],
                    [153, 127, 76]], dtype=np.uint8)


def decode_mask(rgb):
    rgb = np.asarray(rgb)[..., :3]
    labels = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        labels[np.all(rgb == color, axis=-1)] = i
    if np.any(labels == 255):
        unknown = np.unique(rgb[labels == 255], axis=0)
        raise ValueError(f'Unknown mask colors: {unknown[:10].tolist()}')
    return labels


def read_image(path, size=224):
    with Image.open(path) as im:
        # All supplied inputs have equal RGB channels and opaque alpha.
        gray = im.convert('L').resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def apply_transform(tensor, variant):
    # Match the released augment_data.py, including its duplicate 180-degree
    # transforms (both-axes flip and rotate180). 1 original + 5 variants.
    if variant == 0:
        return tensor
    if variant == 1:
        return tensor.flip(-2)
    if variant == 2:
        return tensor.flip(-1)
    if variant in (3, 5):
        return tensor.flip((-2, -1))
    if variant == 4:
        return tensor.rot90(-1, (-2, -1))
    raise ValueError(variant)


class PhaseDataset(Dataset):
    def __init__(self, root, split, size=224, augment=False):
        self.root, self.split, self.size = Path(root), split, size
        if augment and split != 'train':
            raise ValueError('Augmentation is only allowed on train')
        self.variants = 6 if augment else 1
        self.paths = sorted((self.root / split / 'images').glob('*.png'))
        if not self.paths:
            raise ValueError(f'No images in {self.root / split}')
        self.images, self.labels, self.original_labels = [], [], []
        for path in self.paths:
            self.images.append(read_image(path, size))
            if split != 'test':
                mask_path = self.root / split / 'masks' / path.name.replace('_image', '_mask')
                with Image.open(mask_path) as mask, Image.open(path) as im:
                    if mask.size != im.size:
                        raise ValueError(f'Image/mask size mismatch: {path}')
                    original = decode_mask(np.asarray(mask.convert('RGB')))
                small = np.array(Image.fromarray(original).resize(
                    (size, size), Image.Resampling.NEAREST), copy=True)
                self.labels.append(torch.from_numpy(small).long())
                if split == 'valid':
                    self.original_labels.append(original)

    def __len__(self):
        return len(self.paths) * self.variants

    def __getitem__(self, index):
        i, variant = divmod(index, self.variants)
        image = apply_transform(self.images[i], variant)
        if self.split == 'test':
            return image, i
        return image, apply_transform(self.labels[i], variant), i


def _raw_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L'), dtype=np.uint8).copy()


def _patch_origins(height, width, patch, stride):
    ys = list(range(0, max(1, height - patch + 1), stride))
    xs = list(range(0, max(1, width - patch + 1), stride))
    if not ys or ys[-1] != height - patch:
        ys.append(height - patch)
    if not xs or xs[-1] != width - patch:
        xs.append(width - patch)
    return [(y, x) for y in ys for x in xs]


class NativePatchDataset(Dataset):
    """On-the-fly crops from original-resolution Train images and masks.

    Half of draws come from 224x224 tiles with >=10% Eutectic Si pixels;
    remaining draws select any tile. Sources are sampled uniformly before
    choosing a tile so a specimen with many positive tiles cannot dominate.
    Six geometric transforms are sampled online; no data files are generated.
    """
    full_resolution = False

    def __init__(self, root, patch=224, stride=112, eutectic_fraction=0.10,
                 enriched_probability=0.5):
        self.root, self.size = Path(root), patch
        self.paths = sorted((self.root / 'train/images').glob('*.png'))
        if not self.paths:
            raise ValueError(f'No images in {self.root / "train/images"}')
        self.images, self.labels, self.origins = [], [], []
        self.enriched = {}
        self.eutectic_fraction = eutectic_fraction
        self.enriched_probability = enriched_probability
        self.variants = 6
        self.stride = stride
        self.positive_tile_count = 0
        for source_id, path in enumerate(self.paths):
            image = _raw_gray(path)
            mask_path = self.root / 'train/masks' / path.name.replace('_image', '_mask')
            with Image.open(mask_path) as mask:
                label = decode_mask(np.asarray(mask.convert('RGB')))
            if image.shape != label.shape:
                raise ValueError(f'Image/mask shape mismatch: {path}')
            self.images.append(image)
            self.labels.append(label)
            origins = _patch_origins(*image.shape, patch, stride)
            self.origins.append(origins)
            h, w = image.shape
            integral = np.pad((label == 2).astype(np.int64), ((1, 0), (1, 0)))
            integral = integral.cumsum(0).cumsum(1)
            positive = []
            for y, x in origins:
                y2, x2 = y + patch, x + patch
                count = (integral[y2, x2] - integral[y, x2]
                         - integral[y2, x] + integral[y, x])
                if count >= eutectic_fraction * patch * patch:
                    positive.append((y, x))
            if positive:
                self.enriched[source_id] = positive
                self.positive_tile_count += len(positive)
        if not self.enriched:
            raise ValueError('No tile met the Eutectic Si sampling threshold')

    def __len__(self):
        return len(self.paths) * self.variants

    def __getitem__(self, index):
        choose_enriched = torch.rand(()) < self.enriched_probability
        if choose_enriched:
            sources = tuple(self.enriched)
            source = sources[int(torch.randint(len(sources), ()).item())]
            tiles = self.enriched[source]
        else:
            source = int(torch.randint(len(self.paths), ()).item())
            tiles = self.origins[source]
        y, x = tiles[int(torch.randint(len(tiles), ()).item())]
        patch, label = self.size, self.labels[source]
        image_crop = self.images[source][y:y+patch, x:x+patch]
        mask_crop = label[y:y+patch, x:x+patch]
        image = torch.from_numpy(image_crop.copy()).float().unsqueeze(0) / 255.0
        target = torch.from_numpy(mask_crop.astype(np.int64, copy=True))
        # Random element of the dihedral group: 0/90/180/270, optionally flipped.
        k = int(torch.randint(4, ()).item())
        image, target = image.rot90(k, (-2, -1)), target.rot90(k, (-2, -1))
        if bool(torch.randint(2, ()).item()):
            image, target = image.flip(-1), target.flip(-1)
        return image, target, source


class FullResolutionDataset:
    """Full-resolution validation/test images and masks for overlap tiling."""
    full_resolution = True

    def __init__(self, root, split):
        self.root, self.split = Path(root), split
        self.paths = sorted((self.root / split / 'images').glob('*.png'))
        if not self.paths:
            raise ValueError(f'No images in {self.root / split}')
        self.images, self.original_labels = [], []
        for path in self.paths:
            self.images.append(_raw_gray(path))
            if split != 'test':
                mask_path = self.root / split / 'masks' / path.name.replace('_image', '_mask')
                with Image.open(mask_path) as mask:
                    self.original_labels.append(decode_mask(np.asarray(mask.convert('RGB'))))
