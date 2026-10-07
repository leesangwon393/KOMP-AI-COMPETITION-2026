import unittest

import torch

from komap_phase.config import ROOT,load_config
from komap_phase.models import SegmentationModel
from komap_phase.rsu_models import RSUEncoder
from komap_phase import vendor_u2net as upstream


def setUpModule(): torch.set_num_threads(4)


class RSUTest(unittest.TestCase):
    def test_upstream_blocks_preserve_odd_rectangular_shapes_and_backpropagate(self):
        for cls in (upstream.RSU7,upstream.RSU6,upstream.RSU5,upstream.RSU4,upstream.RSU4F):
            block=cls(8,4,16)
            x=torch.randn(2,8,65,67,requires_grad=True)
            y=block(x)
            self.assertEqual(tuple(y.shape),(2,16,65,67))
            y.square().mean().backward()
            self.assertGreater(float(x.grad.abs().sum()),0)
            self.assertTrue(x.grad.isfinite().all())

    def test_backbone_adapters_match_existing_decoder_scales(self):
        c=load_config(ROOT/'configs/U0.json');model=SegmentationModel(c).eval()
        with torch.inference_mode():
            features=model.encoder(torch.rand(1,1,64,64))
            pred=model(torch.rand(1,1,64,64))
        self.assertEqual([tuple(f.shape[1:]) for f in features],
                         [(64,32,32),(256,16,16),(512,8,8),(1024,4,4),(2048,2,2)])
        self.assertEqual(tuple(pred.shape),(1,4,64,64))

    def test_full_no_attention_matches_upstream_probabilities(self):
        c=load_config(ROOT/'configs/F0.json')
        torch.manual_seed(42);reference=upstream.U2NET(3,4).eval()
        rng=torch.get_rng_state().clone()
        torch.manual_seed(42);model=SegmentationModel(c).eval()
        self.assertEqual(set(reference.state_dict()),set(model.full.state_dict()))
        self.assertTrue(all(torch.equal(v,model.full.state_dict()[k]) for k,v in reference.state_dict().items()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(model.full.outconv.in_channels,24)
        x=torch.arange(4096).reshape(1,1,64,64).float()/4096
        z=(x.repeat(1,3,1,1)-model.input_mean)/model.input_std
        with torch.inference_mode():
            expected=reference(z)
            logits,sides=model.full(z)
            self.assertTrue(torch.equal(logits.sigmoid(),expected[0]))
            self.assertTrue(all(torch.equal(a.sigmoid(),b) for a,b in zip(sides,expected[1:])))

    def test_scse_ablation_and_contact_keep_common_initialization(self):
        c=load_config(ROOT/'configs/U0.json')
        torch.manual_seed(42);baseline=SegmentationModel(c).eval()
        state=baseline.state_dict();rng=torch.get_rng_state().clone()
        x=torch.rand(1,1,64,64)
        with torch.inference_mode(): expected=baseline(x)
        for name in ('U1','U2','U3_R02','U3_R03'):
            torch.manual_seed(42);model=SegmentationModel(load_config(ROOT/f'configs/{name}.json')).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            common=set(state)&set(model.state_dict())
            self.assertTrue(all(torch.equal(state[k],model.state_dict()[k]) for k in common))
            if name!='U1':
                with torch.inference_mode(): self.assertTrue(torch.equal(expected,model(x)))

    def test_full_side_supervision_only_changes_loss_not_initial_output(self):
        torch.manual_seed(42);a=SegmentationModel(load_config(ROOT/'configs/F1.json')).eval()
        rng=torch.get_rng_state().clone()
        torch.manual_seed(42);b=SegmentationModel(load_config(ROOT/'configs/F3.json')).eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        x=torch.rand(1,1,64,64)
        with torch.inference_mode(): self.assertTrue(torch.equal(a(x),b(x)))


if __name__=='__main__': unittest.main()
