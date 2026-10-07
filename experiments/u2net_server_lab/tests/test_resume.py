"""Synthetic-fixture epoch transaction/RNG resume; never trains on KoMaP data."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from komap_phase.config import ROOT, load_config
from komap_phase.data import PALETTE
from komap_phase.training import train
import komap_phase.training as training
from run import compare, evaluate_checkpoint


def fixture(root):
    for index,split in enumerate(('train','valid')):
        (root/split/'images').mkdir(parents=True)
        (root/split/'masks').mkdir()
        image=np.random.default_rng(100+index).integers(0,256,(64,64),dtype=np.uint8)
        mask=np.zeros((64,64),dtype=np.uint8)
        mask[:32,32:]=1
        mask[32:,:32]=3
        mask[32:,32:]=2
        Image.fromarray(image).save(root/split/'images'/f'{split}_image.png')
        Image.fromarray(PALETTE[mask]).save(root/split/'masks'/f'{split}_mask.png')


def quick_evaluation(*args,**kwargs):
    return {'miou':.5,'class_iou':[.5]*4,'filenames':['valid_image.png']}


class ResumeTest(unittest.TestCase):
    def test_real_tiling_exports_checkpoint_evaluation_and_comparison(self):
        with tempfile.TemporaryDirectory(prefix='komap_phase_outputs_') as temp:
            root=Path(temp);fixture(root/'data')
            for name in ('B','B_D03'):
                c=load_config(ROOT/f'configs/{name}.json')
                c['model'].update(backbone='resnet34',initialization='none')
                c['train'].update(epochs=1,patch=64,target=32,stride=16,samples_per_epoch=4,batch_size=2,enriched_probability=0.)
                with contextlib.redirect_stdout(io.StringIO()):
                    train(c,root/'data',root/name,torch.device('cpu'))
                summary=json.loads((root/name/'summary.json').read_text())
                self.assertIn('d4',summary)
                self.assertGreater(summary['single']['contact']['pixels'],0)
                self.assertTrue((root/name/'valid_masks/d4/valid_mask.png').is_file())
            with contextlib.redirect_stdout(io.StringIO()):
                compare(SimpleNamespace(runs=[root/'B',root/'B_D03'],output=root/'comparison'))
                evaluate_checkpoint(SimpleNamespace(checkpoint=root/'B/best.pt',data=root/'data',
                    device='cpu',split='valid',d4=False,output=root/'evaluation'))
            self.assertTrue((root/'comparison/comparison.csv').is_file())
            self.assertTrue((root/'evaluation/masks/valid_mask.png').is_file())
            rows=json.loads((root/'comparison/comparison.json').read_text())
            self.assertEqual(len(rows),4)
            self.assertEqual(rows[0]['delta_miou_pp'],0)

    def test_interrupted_run_matches_uninterrupted_with_recurrent_gate_contact(self):
        c=load_config(ROOT/'configs/U3_R02.json')
        c['model'].update(backbone='resnet34',initialization='none')
        c['train'].update(epochs=2,patch=64,target=32,stride=16,samples_per_epoch=4,batch_size=2,enriched_probability=0.)
        c['aux'].update(contact_weight=2.,warmup_epochs=0,ramp_epochs=1)
        c['evaluation']['final_d4']=False
        with tempfile.TemporaryDirectory(prefix='komap_phase_resume_') as temp:
            root=Path(temp);fixture(root/'data')
            with patch.object(training,'evaluate',side_effect=quick_evaluation),contextlib.redirect_stdout(io.StringIO()):
                train(c,root/'data',root/'full',torch.device('cpu'))
            calls=0
            def interrupt_second_validation(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:
                    raise KeyboardInterrupt('synthetic interruption')
                return quick_evaluation()
            with patch.object(training,'evaluate',side_effect=interrupt_second_validation),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    train(c,root/'data',root/'resumed',torch.device('cpu'))
            with patch.object(training,'evaluate',side_effect=quick_evaluation),contextlib.redirect_stdout(io.StringIO()):
                train(c,root/'data',root/'resumed',torch.device('cpu'),resume=True)
            full=torch.load(root/'full/last.pt',weights_only=True)
            resumed=torch.load(root/'resumed/last.pt',weights_only=True)
            self.assertTrue(all(torch.equal(v,resumed['model'][k]) for k,v in full['model'].items()))
            self.assertTrue(torch.equal(full['contrast_rng'],resumed['contrast_rng']))
            self.assertTrue(torch.equal(full['sampling_rng'],resumed['sampling_rng']))
            self.assertTrue(torch.equal(full['rng']['torch'],resumed['rng']['torch']))
            self.assertTrue(torch.equal(full['rng']['loader'],resumed['rng']['loader']))
            self.assertEqual(full['scheduler'],resumed['scheduler'])
            self.assertEqual(full['best_epoch'],resumed['best_epoch'])
            for first,second in zip(full['history'],resumed['history']):
                self.assertEqual(first['train_total_loss'],second['train_total_loss'])


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()


def setUpModule():
    torch.set_num_threads(4)
