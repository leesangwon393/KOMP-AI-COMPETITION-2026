"""Verify actual attention placement, shared initialization, and the ECA residual path."""
import gc
import unittest
import torch
from torch import nn
from komap_phase.config import ROOT, load_config, validate
from komap_phase.models import SegmentationModel
from komap_phase.baseline_models import SCSE
from komap_phase.paper_modules import ECA, CBAM
from komap_phase.runtime import seed_everything


def setUpModule():
    torch.set_num_threads(4)


class U2Attention(unittest.TestCase):
    def test_all_variants_share_base_parameters_and_global_rng(self):
        seed_everything(42)
        base=SegmentationModel(load_config(ROOT/'configs/U2_PLAIN_DEEP.json'),False)
        shared={k:v.clone() for k,v in base.state_dict().items()}
        rng=torch.get_rng_state().clone()
        del base
        for name in ('SCSE_ALL','SCSE_LAST2','ECA','CBAM','ECA_CBAM','ECA_SCSE'):
            seed_everything(42)
            model=SegmentationModel(load_config(ROOT/f'configs/U2_{name}_DEEP.json'),False)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()),name)
            state=model.state_dict()
            self.assertTrue(all(torch.equal(value,state[k]) for k,value in shared.items()),name)
            del model,state
            gc.collect()

    def test_scse_locations_and_cbam_heads(self):
        for name,positions in (('SCSE_ALL',range(5)),('SCSE_LAST2',(3,4))):
            model=SegmentationModel(load_config(ROOT/f'configs/U2_{name}_DEEP.json'),False)
            self.assertEqual([i for i,m in enumerate(model.network.attentions) if isinstance(m,SCSE)],list(positions))
            del model
        model=SegmentationModel(load_config(ROOT/'configs/U2_ECA_CBAM_DEEP.json'),False)
        self.assertEqual(sum(isinstance(m,ECA) for m in model.network.rsu_eca),11)
        self.assertEqual(sum(isinstance(m,CBAM) for m in model.network.fusion_attentions),6)
        self.assertEqual(len(model.network.eca_handles),11)

    def test_eca_preserves_residual_shortcut(self):
        model=SegmentationModel(load_config(ROOT/'configs/U2_ECA_DEEP.json'),False).eval()
        class Zero(nn.Module):
            def forward(self,x):return x*0
        model.network.rsu_eca[0]=Zero()
        stage=model.network.stage1
        image=torch.rand(2,3,64,64)
        with torch.no_grad():
            self.assertTrue(torch.equal(stage(image),stage.rebnconvin(image)))

    def test_ignored_or_mismatched_attention_is_rejected(self):
        c=load_config(ROOT/'configs/U2_PLAIN_DEEP.json')
        c['model']['attention']='scse'
        with self.assertRaisesRegex(ValueError,'mismatch'):validate(c)
        c=load_config(ROOT/'configs/B.json')
        c['model']['u2_rsu_attention']='eca'
        with self.assertRaisesRegex(ValueError,'require'):validate(c)


if __name__=='__main__':unittest.main()
