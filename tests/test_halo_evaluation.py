import unittest
import numpy as np
import torch
from evaluate_context_halo import predict


class PixelModel(torch.nn.Module):
    def forward(self, x):
        return torch.cat([x, -x, 2*x, torch.zeros_like(x)], dim=1)


class HaloTest(unittest.TestCase):
    def test_center_alignment_overlap_and_d4(self):
        torch.set_num_threads(2)
        image = np.random.default_rng(42).integers(0, 256, (239, 277), dtype=np.uint8)
        model = PixelModel().eval()
        expected = model(torch.from_numpy(image.copy()).float()[None, None]/255).softmax(1)[0]
        single = predict(model, image, 'cpu', tta=False)
        d4 = predict(model, image, 'cpu', tta=True)
        for result in [single[0], single[1], d4[0], d4[1]]:
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


if __name__ == '__main__': unittest.main()
