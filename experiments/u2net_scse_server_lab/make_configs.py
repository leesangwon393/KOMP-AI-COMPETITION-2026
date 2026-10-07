"""Dedicated U²-Net attention ablations; isolated from the 127-model release."""
import argparse
import copy
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = [
    ('PLAIN', 'none', 'none', 'none', 'none', 'U²-Net without attention'),
    ('SCSE_ALL', 'scse', 'all', 'none', 'none', 'scSE after all five decoder RSUs'),
    ('SCSE_LAST2', 'scse', 'last2', 'none', 'none', 'scSE after stage2d/stage1d only'),
    ('ECA', 'none', 'none', 'eca', 'none', 'ECA on final residual branch inside all 11 outer RSUs'),
    ('CBAM', 'none', 'none', 'none', 'cbam', 'CBAM on six side-head feature maps before prediction/fusion'),
    ('ECA_CBAM', 'none', 'none', 'eca', 'cbam', 'RSU residual-branch ECA + six pre-fusion CBAM gates'),
    ('ECA_SCSE', 'scse', 'all', 'eca', 'none', 'RSU residual-branch ECA + five decoder scSE gates'),
]


def catalogue():
    base = json.loads((ROOT / 'configs/B.json').read_text())
    records = [base]
    scratch = copy.deepcopy(base)
    scratch.update(id='B_SCRATCH', description='R101 U-Net/scSE baseline, fresh scratch initialization')
    scratch['model']['initialization'] = 'none'
    records.append(scratch)
    # Deep-supervised variants first: the primary comparison, followed by fused-only ablations.
    for deep in (True, False):
        for name, attention, stages, rsu, fusion, description in VARIANTS:
            c = copy.deepcopy(base)
            c.update(id='U2_' + name + ('_DEEP' if deep else ''),
                     description=description + ('; six-side deep supervision' if deep else '; fused loss only'))
            c['model'].update(architecture='u2net', backbone='u2net_full', initialization='none',
                              attention=attention, deep_supervision=deep, u2_decoder_stages=stages,
                              u2_rsu_attention=rsu, u2_fusion_attention=fusion)
            c['aux']['deep_weight'] = .2 if deep else 0.
            c['paper'] = {'source': 'U2NET/scSE/ECA/CBAM', 'implementation': 'derived_attention_ablation',
                          'group': 'u2_attention'}
            records.append(c)
    return records


def generate(output, seed=42, epochs=150, batch_size=4, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for c in catalogue():
        c['train'].update(seed=seed, epochs=epochs, batch_size=batch_size)
        c['evaluation']['batch_size'] = batch_size
        if smoke:
            c['train'].update(patch=64, target=32, stride=16, samples_per_epoch=4, enriched_probability=0.)
            c['model']['initialization'] = 'none'
        payload = json.dumps(c, ensure_ascii=False, indent=2) + '\n'
        path = output / (c['id'] + '.json')
        if path.exists() and path.read_text() != payload:
            raise ValueError('Existing config differs: ' + str(path))
        if not path.exists():
            path.write_text(payload)
    return output


def write_catalogue():
    fields = ['id', 'architecture', 'initialization', 'decoder_scse', 'rsu_attention',
              'fusion_attention', 'deep_weight', 'source', 'implementation', 'description']
    with (ROOT / 'EXPERIMENTS.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in catalogue():
            m = c['model']
            writer.writerow({'id': c['id'], 'architecture': m['architecture'],
                             'initialization': m['initialization'], 'decoder_scse': m['u2_decoder_stages'],
                             'rsu_attention': m['u2_rsu_attention'], 'fusion_attention': m['u2_fusion_attention'],
                             'deep_weight': c['aux']['deep_weight'], 'source': c['paper']['source'],
                             'implementation': c['paper']['implementation'], 'description': c['description']})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'configs')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    generate(a.output, a.seed, a.epochs, a.batch_size)
    write_catalogue()
