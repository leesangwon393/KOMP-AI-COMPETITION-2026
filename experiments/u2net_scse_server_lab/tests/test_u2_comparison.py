"""A U²-Net attention gain must use the paired U²-Net reference, not B."""
import json
import tempfile
import unittest
from pathlib import Path
from komap_phase.config import ROOT, load_config
from summarize_seeds import summarize_u2_pairs


class PairedComparison(unittest.TestCase):
    def test_same_seed_plain_reference_and_missing_seed(self):
        ids=['U2_PLAIN_DEEP','U2_SCSE_ALL_DEEP']
        with tempfile.TemporaryDirectory() as t:
            plan=Path(t)
            for seed in (42,43):
                for name in ids:
                    path=plan/f'seed{seed}'/name;path.mkdir(parents=True)
                    c=load_config(ROOT/f'configs/{name}.json');c['train']['seed']=seed
                    values=[.7,.8] if name==ids[0] else [.71,.83]
                    metric={'miou':sum(values)/len(values),'per_image_miou':values,'filenames':['a','b'],
                            'class_iou':[.8,.7,.5,.9],'contact':{'error_rate':.2},
                            'eutectic':{'thin_skeleton_recall':.6}}
                    files={'config.json':c,'status.json':{'state':'completed'},
                           'summary.json':{'single':metric,'d4':metric},
                           'data_manifest.json':{'valid':['a','b']},'environment.json':{'code':'same'}}
                    for filename,value in files.items():(path/filename).write_text(json.dumps(value))
            rows,details=summarize_u2_pairs(plan,{'ids':ids,'seeds':[42,43,44]})
            candidate=next(r for r in rows if r['id']==ids[1] and r['view']=='single')
            self.assertEqual(candidate['reference_id'],ids[0])
            self.assertEqual(candidate['completed_seed_count'],2)
            self.assertFalse(candidate['all_seeds_completed'])
            self.assertAlmostEqual(candidate['mean_delta_vs_u2_pp'],2.)
            self.assertEqual(candidate['wins_vs_u2'],2)
            self.assertEqual(len(details),8)
            c=load_config(ROOT/f'configs/{ids[1]}.json');c['train']['epochs']=151
            (plan/'seed42'/ids[1]/'config.json').write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'protocol differs'):
                summarize_u2_pairs(plan,{'ids':ids,'seeds':[42]})


if __name__=='__main__':unittest.main()
