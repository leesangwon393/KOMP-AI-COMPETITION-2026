import copy
import json
import tempfile
import unittest
from pathlib import Path

from komap_phase.config import ROOT,load_config,protocol,validate
from make_configs import generate
from server_suite import selection
from summarize_seeds import summarize


class PlanTest(unittest.TestCase):
    def test_explicit_combination_adds_four_controls(self):
        self.assertEqual(selection('validation',None),['B','D03'])
        self.assertEqual(selection('combinations',['R02_D03']),['B','D03','R02','R02_D03'])
        with self.assertRaises(ValueError): selection('combinations',None)
        with self.assertRaises(ValueError): selection('architectures',['R02_D03'])

    def test_generation_preserves_epochs_and_protects_existing_configs(self):
        with tempfile.TemporaryDirectory() as temp:
            output=generate(Path(temp)/'configs',43,150)
            b=load_config(output/'B.json');d=load_config(output/'D03.json')
            self.assertEqual(b['train']['seed'],43)
            self.assertEqual(protocol(b),protocol(d))
            generate(output,43,150)
            with self.assertRaises(ValueError): generate(output,44,150)
            self.assertEqual(load_config(output/'B.json')['train']['seed'],43)

    def test_structures_can_compare_but_loss_budget_changes_cannot(self):
        b=load_config(ROOT/'configs/B.json')
        for name in ('A01_CNX','A02_OS16','A03_R50'):
            self.assertEqual(protocol(b),protocol(load_config(ROOT/f'configs/{name}.json')))
        changed=copy.deepcopy(b);changed['train']['epochs']=200
        self.assertNotEqual(protocol(b),protocol(changed))
        changed=copy.deepcopy(b);changed['model'].update(backbone='convnext_tiny',output_stride=16)
        with self.assertRaises(ValueError): validate(changed)

    def test_seed_summary_uses_paired_differences(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'plan.json').write_text(json.dumps({'ids':['B','D03'],'seeds':[43,44],'epochs':150}))
            for seed,scores in ((43,(.8,.81)),(44,(.82,.80))):
                rows=[]
                for name,score in zip(('B','D03'),scores):
                    path=root/f'seed{seed}'/name;path.mkdir(parents=True)
                    c=load_config(ROOT/f'configs/{name}.json');c['train']['seed']=seed
                    (path/'config.json').write_text(json.dumps(c))
                    (path/'data_manifest.json').write_text('{}')
                    (path/'environment.json').write_text('{}')
                    rows.extend({'id':name,'view':view,'miou':score} for view in ('single','d4'))
                output=root/'comparisons'/f'seed{seed}';output.mkdir(parents=True)
                (output/'comparison.json').write_text(json.dumps(rows))
            result=summarize(root)
            d=next(r for r in result['summary'] if r['id']=='D03' and r['view']=='d4')
            self.assertAlmostEqual(d['mean_delta_vs_B_pp'],-.5)
            self.assertEqual(d['wins_vs_B_seeds'],1)
            self.assertAlmostEqual(d['delta_vs_B_by_seed_pp']['43'],1.)


if __name__=='__main__': unittest.main()
