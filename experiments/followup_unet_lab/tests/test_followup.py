import copy
import unittest
import torch
from torch.nn import functional as F
from komap_phase.config import BASE, validate
from komap_phase.followup_modules import (reassemble, FrequencyFusionNoOffset,
    PixelShuffleUpsample, connectivity_targets, connectivity_loss, DIRECTIONS)
from komap_phase.loss import R022Loss
from experiment_queue import plan, validate_contrasts


class FollowupTests(unittest.TestCase):
    def test_filters_match_unfold_values_and_gradients(self):
        for size,scale in ((3,1),(5,2)):
            feature=torch.randn(2,3,4,5,requires_grad=True)
            logits=torch.randn(2,size*size,4*scale,5*scale,requires_grad=True)
            mask=logits.softmax(1)
            actual=reassemble(feature,mask,size,scale)
            unfolded=F.unfold(F.pad(feature,(size//2,)*4,mode='reflect'),size)
            unfolded=unfolded.reshape(2,3*size*size,4,5)
            expected=(F.interpolate(unfolded,scale_factor=scale,mode='nearest').reshape(2,3,size*size,4*scale,5*scale)*mask[:,None]).sum(2)
            torch.testing.assert_close(actual,expected)
            first=torch.autograd.grad(actual.square().sum(),(feature,logits),retain_graph=True)
            second=torch.autograd.grad(expected.square().sum(),(feature,logits))
            for a,b in zip(first,second):torch.testing.assert_close(a,b)

    def test_frequency_preserves_constant_and_kernels_sum_to_one(self):
        module=FrequencyFusionNoOffset(3,3,4).eval()
        high,low=module(torch.ones(2,3,8,10),torch.ones(2,3,4,5))
        torch.testing.assert_close(high,torch.ones_like(high))
        torch.testing.assert_close(low,torch.ones_like(low))
        masks=module.normalize(torch.randn(2,25,8,10),5)
        torch.testing.assert_close(masks.sum(1),torch.ones(2,8,10))

    def test_connectivity_direction_class0_and_no_wrap(self):
        target=torch.tensor([[[0,0,1],[0,2,2],[3,3,2]]])
        desired,valid=connectivity_targets(target)
        right=DIRECTIONS.index((0,1));left=DIRECTIONS.index((0,-1))
        self.assertEqual(desired[0,right,0,0].item(),1)
        self.assertEqual(desired[0,right,0,1].item(),0)
        self.assertFalse(valid[0,right,0,2].item())
        self.assertFalse(valid[0,left,0,0].item())
        self.assertEqual(desired[0,2*8+right,1,1].item(),1)

    def test_invalid_border_has_zero_gradient(self):
        target=torch.zeros(2,4,4,dtype=torch.long)
        logits=torch.randn(2,32,4,4,requires_grad=True)
        loss=connectivity_loss(logits,target);loss.backward()
        _,valid=connectivity_targets(target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(logits.grad[~valid].abs().sum().item(),0)
        self.assertGreater(logits.grad[:,0:8][valid[:,0:8]].abs().sum().item(),0)

    def test_pixelshuffle_equal_initial_subpixels(self):
        result=PixelShuffleUpsample(3)(torch.randn(2,3,4,5))
        self.assertEqual(result.shape,(2,3,8,10))
        for dy,dx in ((0,1),(1,0),(1,1)):
            torch.testing.assert_close(result[:,:,::2,::2],result[:,:,dy::2,dx::2])

    def test_loss_ratios_match_weighted_components(self):
        logits=torch.randn(2,4,8,8);target=torch.randint(4,(2,8,8))
        default=R022Loss()(logits,target)
        a=R022Loss(.5,.35,.15)(logits,target)
        b=R022Loss(.5,.15,.35)(logits,target)
        torch.testing.assert_close((a+b)/2,default)

    def test_plan_exactly_thirteen_and_one_factor(self):
        self.assertEqual(sum(len(group['ids']) for group in plan()),13)
        self.assertEqual(set(validate_contrasts()),set(plan()[2]['ids'][1:]))
        for group in plan():self.assertEqual(group['ids'][0],'B')

    def test_reject_mismatched_head_and_loss(self):
        config=copy.deepcopy(BASE);config['model']['connectivity']=True
        with self.assertRaises(ValueError):validate(config)


if __name__=='__main__':unittest.main()
