import json
import tempfile
import unittest
from pathlib import Path

import torch

from komap_phase.config import ROOT,load_config,protocol,validate
from komap_phase.data import MatchedPatchDataset
from komap_phase.objectives import PhaseLoss,contact_band,weighted_contact_ce
from make_configs import generate
from server_suite import selection
from summarize_seeds import summarize


def setUpModule(): torch.set_num_threads(4)


class ControlTest(unittest.TestCase):
    def test_stage_controls_and_config_protection(self):
        self.assertEqual(selection('core',None),['B','B_D03','U0','U1','U2'])
        self.assertEqual(selection('rrcu',['U3_R02']),['U0','U3_R02'])
        self.assertEqual(selection('deep',['F3']),['F1','F3'])
        with self.assertRaises(ValueError): selection('core',['F3'])
        with tempfile.TemporaryDirectory() as temp:
            path=generate(Path(temp)/'configs',43,150)
            self.assertEqual(load_config(path/'U0.json')['train']['seed'],43)
            generate(path,43,150)
            with self.assertRaises(ValueError): generate(path,44,150)

    def test_declared_system_changes_compare_but_budget_changes_do_not(self):
        b=load_config(ROOT/'configs/B.json')
        for path in (ROOT/'configs').glob('*.json'):
            self.assertEqual(protocol(b),protocol(load_config(path)))
        bad=load_config(ROOT/'configs/U0.json');bad['model']['initialization']='imagenet'
        with self.assertRaises(ValueError): validate(bad)
        bad=load_config(ROOT/'configs/B.json');bad['train']['epochs']=200
        self.assertNotEqual(protocol(b),protocol(bad))

    def test_contact_does_not_wrap_or_weight_background(self):
        y=torch.full((1,8,8),3,dtype=torch.long);y[:,:,0]=1;y[:,:,-1]=2
        self.assertFalse(contact_band(y,2).any())
        y[:,2:6,1:4]=1;y[:,2:6,4:7]=2
        self.assertFalse(contact_band(y,2)[y==3].any())
        logits=torch.zeros(1,4,8,8)
        a,_=weighted_contact_ce(logits,y,2.,2)
        self.assertAlmostEqual(float(a),float(torch.nn.functional.cross_entropy(logits,y)),places=6)

    def test_deep_supervision_is_normalized_and_has_side_gradient(self):
        c=load_config(ROOT/'configs/F3.json');criterion=PhaseLoss(c)
        x=torch.randn(2,4,32,32,requires_grad=True);y=torch.randint(4,(2,32,32))
        sides=[x.detach().clone().requires_grad_() for _ in range(6)]
        total,parts=criterion(x,y,sides,1)
        self.assertTrue(torch.allclose(total,criterion.base(x,y),atol=1e-6))
        total.backward()
        self.assertTrue(all(s.grad is not None and s.grad.abs().sum()>0 for s in sides))
        self.assertGreater(parts['deep_supervision_loss'],0)
        with self.assertRaises(ValueError): criterion(x,y,sides[:5],1)

    def test_sampling_is_independent_of_global_model_rng_and_resumable(self):
        from test_resume import fixture
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            dataset=MatchedPatchDataset(root,context=64,target=32,stride=16,samples_per_epoch=4,seed=42)
            state=dataset.draw_rng.get_state()
            global_state=torch.get_rng_state().clone()
            a=dataset[0]
            self.assertTrue(torch.equal(global_state,torch.get_rng_state()))
            dataset.draw_rng.set_state(state)
            torch.rand(100000)
            b=dataset[0]
            self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(a[1],b[1]))

    def test_seed_summary_uses_selected_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'plan.json').write_text(json.dumps({'ids':['F1','F3'],'seeds':[43,44],'epochs':150}))
            for seed,scores in ((43,(.8,.81)),(44,(.82,.80))):
                rows=[]
                for name,score in zip(('F1','F3'),scores):
                    path=root/f'seed{seed}'/name;path.mkdir(parents=True)
                    c=load_config(ROOT/f'configs/{name}.json');c['train']['seed']=seed
                    (path/'config.json').write_text(json.dumps(c))
                    (path/'data_manifest.json').write_text('{}');(path/'environment.json').write_text('{}')
                    rows.extend({'id':name,'view':view,'miou':score} for view in ('single','d4'))
                path=root/'comparisons'/f'seed{seed}';path.mkdir(parents=True)
                (path/'comparison.json').write_text(json.dumps(rows))
            report=summarize(root)
            self.assertEqual(report['reference_id'],'F1')
            row=next(v for v in report['summary'] if v['id']=='F3' and v['view']=='d4')
            self.assertAlmostEqual(row['mean_delta_vs_reference_pp'],-.5)


if __name__=='__main__': unittest.main()
