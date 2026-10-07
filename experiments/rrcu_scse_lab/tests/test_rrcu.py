import unittest

import torch

from komap_phase.config import ROOT, load_config, protocol
from komap_phase.models import SegmentationModel


def setUpModule():
    torch.set_num_threads(4)


class RRCUTest(unittest.TestCase):
    def test_common_initial_weights_rng_and_gate_identity(self):
        c=load_config(ROOT/'configs/B.json')
        c['model']['backbone']='resnet34'
        torch.manual_seed(42)
        baseline=SegmentationModel(c,False).eval()
        state=baseline.state_dict()
        rng=torch.get_rng_state().clone()
        image=torch.arange(4096).float().reshape(1,1,64,64)/4096
        with torch.inference_mode(): expected=baseline(image)
        for name in ('R01','R02','R03'):
            candidate=load_config(ROOT/f'configs/{name}.json')
            candidate['model']['backbone']='resnet34'
            self.assertEqual(protocol(c),protocol(candidate))
            torch.manual_seed(42)
            model=SegmentationModel(candidate,False).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            common=set(state)&set(model.state_dict())
            self.assertTrue(all(torch.equal(state[k],model.state_dict()[k]) for k in common))
            if name!='R01':
                self.assertEqual(len(common),len(state))
                with torch.inference_mode(): self.assertTrue(torch.equal(expected,model(image)))

    def test_shared_recurrence_and_channel_geometry(self):
        for name in ('R01','R02','R03'):
            c=load_config(ROOT/f'configs/{name}.json');c['model']['backbone']='resnet34'
            model=SegmentationModel(c,False).eval()
            unit=model.decoder.blocks[3] if name=='R01' else model.decoder.blocks[3].attention.recurrent
            calls=[]
            handle=unit.recurrent_conv.register_forward_hook(lambda m,args,output:calls.append(tuple(args[0].shape)))
            with torch.inference_mode(): model(torch.rand(2,1,64,64))
            handle.remove()
            self.assertEqual(len(calls),2)
            if name=='R03':
                self.assertEqual(calls[0][-2:],(1,1))
                self.assertEqual(unit.recurrent_conv.kernel_size,(1,1))

    def test_training_center_and_finite_gradients(self):
        for name in ('R01','R02','R03'):
            c=load_config(ROOT/f'configs/{name}.json')
            c['model']['backbone']='resnet34';c['train'].update(patch=64,target=32,stride=16)
            torch.manual_seed(42)
            model=SegmentationModel(c,False).train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
            image=torch.rand(2,1,64,64);target=torch.randint(4,(2,32,32))
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                logits=model(image)
                self.assertEqual(tuple(logits.shape),(2,4,32,32))
                loss=torch.nn.functional.cross_entropy(logits,target);loss.backward()
                self.assertTrue(all(p.grad.isfinite().all() for p in model.parameters() if p.grad is not None))
                unit=model.decoder.blocks[3] if name=='R01' else model.decoder.blocks[3].attention.recurrent
                module=unit.recurrent_conv if name=='R01' or step==1 else unit.output
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
                optimizer.step()
