import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mimu.contact_sampling import ContactPatchDataset, contact_coordinates
from mimu.data import NativePatchDataset, PALETTE


class ContactSamplingTest(unittest.TestCase):
    def test_contact_excludes_third_class_gap(self):
        label = np.array([[1, 3, 2], [1, 3, 2], [1, 3, 2]])
        self.assertEqual(len(contact_coordinates(label)), 0)
        label[:, 1] = 2
        self.assertEqual(set(map(tuple, contact_coordinates(label))),
                         {(y, x) for y in range(3) for x in (0, 1)})

    def test_sampling_and_baseline_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for d in ['train/images', 'train/masks']:
                (root/d).mkdir(parents=True)
            label = np.ones((32, 32), dtype=np.uint8)
            label[:, 16:] = 2
            Image.fromarray(np.arange(1024).reshape(32, 32).astype(np.uint8)).save(root/'train/images/a_image.png')
            Image.fromarray(PALETTE[label]).save(root/'train/masks/a_mask.png')
            base = NativePatchDataset(root, patch=8, stride=4)
            control = ContactPatchDataset(root, patch=8, stride=4, contact_probability=0)
            torch.manual_seed(42)
            expected = [base[i] for i in range(40)]
            torch.manual_seed(42)
            actual = [control[i] for i in range(40)]
            for a, b in zip(expected, actual):
                self.assertTrue(torch.equal(a[0], b[0]))
                self.assertTrue(torch.equal(a[1], b[1]))
                self.assertEqual(a[2], b[2])
            changed = ContactPatchDataset(root, patch=8, stride=4, contact_probability=.25)
            counts = dict(rich=0, contact=0, all=0)
            torch.manual_seed(123)
            for _ in range(10000):
                source, y, x, route = changed.sample_origin()
                counts[route] += 1
                self.assertTrue(0 <= y <= 24 and 0 <= x <= 24)
                if route == 'contact':
                    crop = changed.labels[source][y:y+8, x:x+8]
                    self.assertTrue(np.any(crop == 1) and np.any(crop == 2))
            for route, probability in [('rich', .5), ('contact', .25), ('all', .25)]:
                self.assertLess(abs(counts[route]/10000-probability), .025)


if __name__ == '__main__':
    unittest.main()
