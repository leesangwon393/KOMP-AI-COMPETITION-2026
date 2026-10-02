import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from mimu.domain_ssl import nt_xent, TrainViews, transfer_encoder, ENCODER_PREFIXES
from mimu.upsampling_variants import UpsamplingUNet


class DomainSSLTests(unittest.TestCase):
    def test_contrastive_targets_and_gradients(self):
        torch.set_num_threads(2)
        a = torch.randn(3, 7, requires_grad=True)
        b = torch.randn(3, 7, requires_grad=True)
        z = F.normalize(torch.cat((a, b)), dim=1)
        logits = z @ z.T/.07
        expected = []
        for i in range(6):
            other = torch.tensor([j for j in range(6) if j != i])
            expected.append(-logits[i,(i+3)%6]+torch.logsumexp(logits[i,other],dim=0))
        loss = nt_xent(a,b)
        torch.testing.assert_close(loss,torch.stack(expected).mean())
        torch.testing.assert_close(loss,nt_xent(b,a))
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())
        self.assertAlmostEqual(nt_xent(torch.ones(4,8),torch.ones(4,8)).item(),math.log(7),places=5)
        matched = torch.eye(4)
        self.assertLess(nt_xent(matched,matched).item(),.001)

    def test_encoder_transfer_preserves_decoder_and_rng(self):
        torch.manual_seed(42)
        model = UpsamplingUNet('dysample',pretrained=False)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        state = {k:v.clone() for k,v in before.items() if k.startswith(ENCODER_PREFIXES)}
        state['conv1.weight'] += .01
        state['bn1.running_mean'] += .1
        rng = torch.get_rng_state().clone()
        transfer_encoder(model,state)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,tensor in before.items():
            torch.testing.assert_close(model.state_dict()[key],state[key] if key in state else tensor)
        with self.assertRaises(ValueError):
            transfer_encoder(model,{k:v for k,v in state.items() if k != 'layer4.0.conv1.weight'})
        with self.assertRaises(ValueError):
            transfer_encoder(model,{**state,'head.weight':before['head.weight']})

    def test_unlabeled_train_only_and_distinct_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root/'train/images'
            images.mkdir(parents=True)
            # No train masks and intentionally unusable validation file.
            valid = root/'valid/images'
            valid.mkdir(parents=True)
            (valid/'bad.png').write_bytes(b'not an image')
            for i in range(4):
                Image.fromarray(np.full((256,256),i*60,dtype=np.uint8)).save(images/f'{i}.png')
            ds = TrainViews(root)
            torch.manual_seed(7)
            a,b,sources = ds.batch(4)
            self.assertEqual(a.shape,(4,1,224,224))
            self.assertEqual(b.shape,a.shape)
            self.assertEqual(len(set(sources.tolist())),4)
            self.assertTrue(all('train' in p.parts for p in ds.paths))
            self.assertTrue(torch.isfinite(a).all() and 0 <= a.min() <= a.max() <= 1)
            rng = torch.get_rng_state()
            expected = ds.batch(4)
            torch.set_rng_state(rng)
            actual = ds.batch(4)
            for x,y in zip(expected,actual):
                self.assertTrue(torch.equal(x,y))


if __name__ == '__main__':
    unittest.main()
