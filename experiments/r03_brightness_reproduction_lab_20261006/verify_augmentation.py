"""Check exact baseline equivalence, paired crops, and complete augmentation RNG resume."""
import hashlib
import json
import sys
from pathlib import Path

import torch

from komap_phase.augmentations import AugmentedNativePatchDataset
from komap_phase.config import ROOT,load_config
from komap_phase.data import NativePatchDataset

def digest(x):return hashlib.sha256(x.numpy().tobytes()).hexdigest()
def main():
    torch.set_num_threads(2)
    path=Path(sys.argv[1]);output=Path(sys.argv[2])
    configs=[load_config(p) for p in sorted((ROOT/'configs').glob('A*.json'))]
    base=configs[0]['augmentation']
    original=NativePatchDataset(path)
    candidate=AugmentedNativePatchDataset.__new__(AugmentedNativePatchDataset)
    candidate.__dict__.update(original.__dict__)
    candidate.extra_generator=torch.Generator().manual_seed(12047)
    candidate.augmentation=dict(base)
    torch.manual_seed(42)
    expected=[original[i] for i in range(16)]
    expected_rng=torch.get_rng_state().clone()
    torch.manual_seed(42)
    actual=[candidate[i] for i in range(16)]
    assert torch.equal(expected_rng,torch.get_rng_state()),'Legacy RNG consumption changed'
    assert all(torch.equal(x[0],y[0]) and torch.equal(x[1],y[1]) and x[2]==y[2] for x,y in zip(expected,actual)), 'Baseline recipe changed'
    reference=[(s,digest(y)) for _,y,s in expected]
    checks=[]
    for c in configs:
        candidate.augmentation=c['augmentation']
        candidate.extra_generator.manual_seed(12047)
        torch.manual_seed(42)
        samples=[candidate[i] for i in range(16)]
        assert [(s,digest(y)) for _,y,s in samples]==reference,'Unpaired crop/geometry: '+c['id']
        assert torch.equal(expected_rng,torch.get_rng_state()),'Global RNG differs: '+c['id']
        assert all(x.shape==(1,448,448) and y.shape==(224,224) and torch.isfinite(x).all() and 0<=x.min()<=x.max()<=1 for x,y,_ in samples)
        # Restoring both RNGs must recover the exact subsequent augmented sample.
        global_rng=torch.get_rng_state().clone();extra_rng=candidate.extra_generator.get_state().clone()
        after=candidate[17]
        torch.set_rng_state(global_rng);candidate.extra_generator.set_state(extra_rng)
        replay=candidate[17]
        assert torch.equal(after[0],replay[0]) and torch.equal(after[1],replay[1]) and after[2]==replay[2]
        checks.append({'id':c['id'],'paired_crops':True,'legacy_rng_equal':True,'resume_rng_equal':True})
    candidate.augmentation=next(c['augmentation'] for c in configs if c['id']=='A08_PHOTO_OFF')
    torch.manual_seed(7)
    raw=torch.rand(1,64,64)
    assert torch.equal(raw,candidate._photometric(raw.clone())),'PHOTO_OFF changes intensity'
    image=torch.full((1,64,64),.5)
    assert torch.allclose(candidate.blur(image,.6),image,atol=1e-6),'Blur changes constant image'
    result={'passed':True,'baseline_bitwise_equal':True,'samples_per_condition':16,'conditions':checks,
            'original_image_count':len(original.paths),'note':'Dataset/RNG verification; no quality claim'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
