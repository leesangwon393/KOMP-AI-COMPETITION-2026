"""Meaningful comparison contracts, reference identity and partial-result handling."""
import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import torch
from komap_phase.config import ROOT,load_config,protocol,validate
from komap_phase.models import SegmentationModel
from komap_phase.baseline_models import SegmentationModel as Reference
from komap_phase.runtime import seed_everything
from make_configs import catalogue
from server_suite import PROFILES
from summarize_seeds import summarize

def setUpModule():torch.set_num_threads(4)

class Contracts(unittest.TestCase):
    def test_catalogue_protocol_and_profiles(self):
        records=catalogue();ids=[c['id'] for c in records]
        self.assertEqual(len(ids),len(set(ids)))
        for c in records:
            validate(c);self.assertEqual(protocol(c),protocol(records[0]))
            self.assertEqual(load_config(ROOT/'configs'/(c['id']+'.json')),c)
        for profile in PROFILES.values():self.assertTrue(set(profile)<=set(ids))

    def test_baseline_parameters_outputs_and_rng_equal_reference(self):
        c=load_config(ROOT/'configs/B.json');c['train'].update(patch=64,target=32,stride=16)
        seed_everything(42);old=Reference(c,False);old_rng=torch.get_rng_state().clone()
        seed_everything(42);new=SegmentationModel(c,False)
        self.assertTrue(torch.equal(old_rng,torch.get_rng_state()))
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        self.assertTrue(all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()))
        image=torch.rand(2,1,64,64);old.eval();new.eval()
        with torch.no_grad():self.assertTrue(torch.equal(old(image),new(image)))

    def test_auxiliary_head_loss_contract_is_rejected(self):
        c=load_config(ROOT/'configs/M_DEEP.json');c['aux']['deep_weight']=0.
        with self.assertRaisesRegex(ValueError,'mismatch'):validate(c)
        c=load_config(ROOT/'configs/F_U2.json');c['model']['initialization']='imagenet'
        with self.assertRaisesRegex(ValueError,'scratch'):validate(c)

    def test_protocol_keeps_epoch_seed_batch_and_loss(self):
        base=load_config(ROOT/'configs/B.json')
        for key in ('seed','epochs','batch_size','lr'):
            altered=copy.deepcopy(base);altered['train'][key]+=1
            self.assertNotEqual(protocol(base),protocol(altered))
        altered=copy.deepcopy(base);altered['model']['backbone']='resnet34'
        self.assertEqual(protocol(base),protocol(altered))

    def test_missing_runs_not_ranked_as_completed(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'plan.json').write_text(json.dumps({'ids':['B','M_OCR'],'seeds':[42,43]}))
            with contextlib.redirect_stdout(io.StringIO()):summarize(root)
            self.assertEqual(json.loads((root/'leaderboard.json').read_text()),[])
            failures=json.loads((root/'failures.json').read_text());self.assertEqual(len(failures),4)

    def test_patch_sampling_is_independent_of_model_rng(self):
        from komap_phase.data import NativePatchDataset
        from test_resume import fixture
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root)
            a=NativePatchDataset(root,context=64,target=32,stride=16,enriched_probability=0.,samples_per_epoch=4,generator=torch.Generator().manual_seed(8150))
            b=NativePatchDataset(root,context=64,target=32,stride=16,enriched_probability=0.,samples_per_epoch=4,generator=torch.Generator().manual_seed(8150))
            for i in range(4):
                first=a[i];torch.rand(109+i);state=torch.get_rng_state().clone();second=b[i]
                self.assertTrue(torch.equal(state,torch.get_rng_state()))
                self.assertTrue(torch.equal(first[0],second[0]));self.assertTrue(torch.equal(first[1],second[1]))
            self.assertTrue(torch.equal(a.generator.get_state(),b.generator.get_state()))

if __name__=='__main__':unittest.main()
