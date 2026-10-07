import copy
import unittest

import torch

from komap_phase.config import load_config, ROOT, validate
from komap_phase.loss import R022Loss
from komap_phase.models import SegmentationModel
from komap_phase.objectives import PhaseLoss, balanced_contrast, contact_band, pure_labels, weighted_contact_ce


class ObjectivesTest(unittest.TestCase):
    def test_contact_no_wrap_or_al_gap(self):
        target = torch.full((1,8,8),3,dtype=torch.long)
        target[:,:,0]=1
        target[:,:,-1]=2
        self.assertFalse(contact_band(target,2).any())
        target[:,:,:4]=1
        target[:,:,4:]=2
        band=contact_band(target,0)
        self.assertEqual(int(band.sum()),16)
        self.assertTrue(band[:,:,3:5].all())

    def test_contact_weighting_only_target_classes(self):
        target=torch.full((1,8,8),3,dtype=torch.long)
        target[:,2:6,1:4]=1
        target[:,2:6,4:7]=2
        band=contact_band(target,2)
        self.assertFalse(band[target==3].any())
        logits=torch.zeros(1,4,8,8)
        weighted,_=weighted_contact_ce(logits,target,2.,2)
        self.assertAlmostEqual(float(weighted),float(torch.nn.functional.cross_entropy(logits,target)),places=6)
        logits[:,:,2:6,3:5]=torch.tensor([5.,0.,0.,0.])[None,:,None,None]
        weighted,_=weighted_contact_ce(logits,target,2.,2)
        self.assertGreater(float(weighted),float(torch.nn.functional.cross_entropy(logits,target)))

    def test_mixed_cells_not_assigned_a_false_class(self):
        target=torch.tensor([[[0,1],[1,0]]])
        labels,pure=pure_labels(target,(1,1),.9)
        self.assertFalse(pure.any())

    def test_separated_embeddings_better_and_finite_gradients(self):
        target=torch.tensor([[[0,0,1,1],[0,0,1,1],[2,2,3,3],[2,2,3,3]]])
        good=torch.nn.functional.one_hot(target,4).permute(0,3,1,2).float().requires_grad_()
        bad=torch.ones_like(good,requires_grad=True)
        a,stats=balanced_contrast(good,target,torch.Generator().manual_seed(42),32,.9,.1)
        b,_=balanced_contrast(bad,target,torch.Generator().manual_seed(42),32,.9,.1)
        self.assertEqual(stats['classes'],4)
        self.assertLess(float(a.detach()),float(b.detach()))
        a.backward()
        self.assertTrue(torch.isfinite(good.grad).all())
        self.assertGreater(float(good.grad.abs().sum()),0)

    def test_empty_contrast_is_differentiable_zero(self):
        x=torch.randn(1,8,4,4,requires_grad=True)
        y=torch.zeros(1,8,8,dtype=torch.long)
        loss,stats=balanced_contrast(x,y,torch.Generator().manual_seed(2))
        self.assertEqual(float(loss.detach()),0)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(stats['points'],0)

    def test_baseline_loss_exact_and_sampling_rng_isolated(self):
        c=load_config(ROOT/'configs/B.json')
        logits=torch.randn(2,4,8,8,requires_grad=True)
        target=torch.randint(4,(2,8,8))
        expected=R022Loss(**c['loss'])(logits,target)
        actual,_=PhaseLoss(c)(logits,target,[],1)
        self.assertTrue(torch.equal(expected,actual))
        x=torch.randn(2,8,8,8)
        state=torch.get_rng_state().clone()
        balanced_contrast(x,target,torch.Generator().manual_seed(42),4,.9,.1)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_auxiliary_allocation_preserves_weights_rng_and_initial_logits(self):
        c=load_config(ROOT/'configs/B.json')
        c['model']['backbone']='resnet34'
        torch.manual_seed(42)
        baseline=SegmentationModel(c,False).eval()
        baseline_rng=torch.get_rng_state().clone()
        image=torch.arange(4096).reshape(1,1,64,64).float()/4096
        with torch.no_grad():
            expected=baseline(image)
        for name in ('D01','D02','D04'):
            candidate=load_config(ROOT/f'configs/{name}.json')
            candidate['model']['backbone']='resnet34'
            torch.manual_seed(42)
            model=SegmentationModel(candidate,False).eval()
            self.assertTrue(torch.equal(baseline_rng,torch.get_rng_state()))
            state=model.state_dict()
            self.assertTrue(all(torch.equal(v,state[k]) for k,v in baseline.state_dict().items()))
            with torch.no_grad():
                self.assertTrue(torch.equal(expected,model(image)))

    def test_warmup_and_schema(self):
        c=load_config(ROOT/'configs/D01.json')
        criterion=PhaseLoss(c)
        logits=torch.randn(2,4,8,8)
        y=torch.randint(4,(2,8,8))
        _,s=criterion(logits,y,[],5)
        self.assertEqual(s['contrast_lambda'],0)
        _,s=criterion(logits,y,[],6)
        self.assertAlmostEqual(s['contrast_lambda'],.01)
        _,s=criterion(logits,y,[],10)
        self.assertAlmostEqual(s['contrast_lambda'],.05)
        c['aux']['purity']=.5
        with self.assertRaises(ValueError):
            validate(c)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()


def setUpModule():
    torch.set_num_threads(4)
