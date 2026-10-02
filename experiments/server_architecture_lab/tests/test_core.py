import unittest
import numpy as np
import torch

from mimu.data import PALETTE, apply_transform, decode_mask
from mimu.loss import HybridLoss
from mimu.metrics import confusion_matrix, summarize
from mimu.model import MIMUNet, RRCNNBlock, AuthorSCSE


class CoreTests(unittest.TestCase):
    def test_palette_roundtrip_and_reject_unknown(self):
        ids = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(decode_mask(PALETTE[ids]), ids)
        bad = PALETTE[ids].copy()
        bad[0, 0] = [255, 255, 255]
        with self.assertRaises(ValueError):
            decode_mask(bad)

    def test_paired_geometry(self):
        mask = torch.arange(16).reshape(4, 4)
        for i in range(6):
            self.assertTrue(torch.equal(apply_transform(mask[None], i)[0], apply_transform(mask, i)))
        self.assertTrue(torch.equal(apply_transform(mask, 3), apply_transform(mask, 5)))

    def test_image_average_not_pixel_pooling(self):
        small = confusion_matrix(np.array([[0, 1], [2, 3]]), np.array([[0, 1], [2, 3]]))
        big = confusion_matrix(np.tile(np.array([[0, 1], [2, 3]]), (10, 10)),
                               np.zeros((20, 20), dtype=int))
        self.assertAlmostEqual(summarize([small, big])['miou'], (1 + 0.25 / 4) / 2)

    def test_absent_policy(self):
        cm = confusion_matrix(np.zeros((2, 2), dtype=int), np.zeros((2, 2), dtype=int))
        self.assertEqual(summarize([cm], 'one')['miou'], 1)
        self.assertEqual(summarize([cm], 'zero')['miou'], .25)
        self.assertEqual(summarize([cm], 'ignore')['miou'], 1)

    def test_graph_and_backward(self):
        torch.set_num_threads(2)
        model = MIMUNet()
        self.assertEqual(sum(isinstance(m, RRCNNBlock) for m in model.modules()), 18)
        self.assertEqual(sum(isinstance(m, AuthorSCSE) for m in model.modules()), 7)
        x = torch.rand(2, 1, 32, 32)
        out = model(x)
        self.assertEqual(tuple(out.shape), (2, 4, 32, 32))
        loss = HybridLoss()(out, torch.randint(0, 4, (2, 32, 32)))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        print('MIMU-Net trainable parameters:', sum(p.numel() for p in model.parameters()))

    def test_loss_rewards_correct_prediction(self):
        target = torch.tensor([[[0, 1], [2, 3]]])
        perfect = torch.nn.functional.one_hot(target, 4).permute(0, 3, 1, 2).float() * 20
        self.assertLess(HybridLoss()(perfect, target), 1e-5)
        self.assertGreater(HybridLoss()(torch.zeros_like(perfect), target), 1)


if __name__ == '__main__':
    unittest.main()
