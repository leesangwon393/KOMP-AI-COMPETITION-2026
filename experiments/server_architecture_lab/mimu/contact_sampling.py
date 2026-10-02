"""Train-only Al3Ni/Eutectic contact-centered patch sampling."""
import numpy as np
import torch
from scipy import ndimage as ndi
from mimu.data import NativePatchDataset


def contact_coordinates(label):
    # Only endpoints of an actual 8-neighbor class-1/class-2 contact.
    a, b = label == 1, label == 2
    footprint = np.ones((3, 3), dtype=bool)
    contact = (a & ndi.binary_dilation(b, structure=footprint)) | (b & ndi.binary_dilation(a, structure=footprint))
    return np.argwhere(contact).astype(np.int32)


class ContactPatchDataset(NativePatchDataset):
    def __init__(self, *args, contact_probability=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= contact_probability <= 1-self.enriched_probability:
            raise ValueError('Invalid contact probability')
        self.contact_probability = contact_probability
        self.contacts = {i: c for i, label in enumerate(self.labels)
                         if len(c := contact_coordinates(label))}
        if contact_probability and not self.contacts:
            raise ValueError('No Al3Ni/Eutectic contact in training data')

    def sample_origin(self):
        draw = float(torch.rand(()))
        if draw < self.enriched_probability:
            sources = tuple(self.enriched)
            source = sources[int(torch.randint(len(sources), ()))]
            tiles = self.enriched[source]
            y, x = tiles[int(torch.randint(len(tiles), ()))]
            route = 'rich'
        elif draw < self.enriched_probability+self.contact_probability:
            sources = tuple(self.contacts)
            source = sources[int(torch.randint(len(sources), ()))]
            coords = self.contacts[source]
            cy, cx = coords[int(torch.randint(len(coords), ()))]
            h, w = self.labels[source].shape
            y = int(np.clip(cy-self.size//2, 0, h-self.size))
            x = int(np.clip(cx-self.size//2, 0, w-self.size))
            route = 'contact'
        else:
            source = int(torch.randint(len(self.paths), ()))
            tiles = self.origins[source]
            y, x = tiles[int(torch.randint(len(tiles), ()))]
            route = 'all'
        return source, y, x, route

    def __getitem__(self, index):
        source, y, x, _ = self.sample_origin()
        p = self.size
        image = torch.from_numpy(self.images[source][y:y+p, x:x+p].copy()).float().unsqueeze(0)/255.
        target = torch.from_numpy(self.labels[source][y:y+p, x:x+p].astype(np.int64, copy=True))
        k = int(torch.randint(4, ()))
        image, target = image.rot90(k, (-2, -1)), target.rot90(k, (-2, -1))
        if bool(torch.randint(2, ())):
            image, target = image.flip(-1), target.flip(-1)
        return image, target, source
