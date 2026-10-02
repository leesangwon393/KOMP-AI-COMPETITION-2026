"""Regenerate complete editable JSON configs; DOES NOT launch any experiment."""
import json
from pathlib import Path
from komap_lab.config import new_config

EXPERIMENTS = [
    ('B', 'R022 recipe baseline: ResNet101 Context448, central224, bilinear', {}),
    ('C01', 'Baseline + shallow HF residual', {'hf': True}),
    ('C02', 'Baseline + layer3-query/layer4-KV attention', {'cross': 'attention'}),
    ('C03', 'Baseline + HF + cross attention', {'hf': True, 'cross': 'attention'}),
    ('C04', 'Adaptive dilation 1/2/3 replaces one decoder scSE', {'dilation': 'adaptive'}),
    ('C05', 'Adaptive dilation + HF', {'dilation': 'adaptive', 'hf': True}),
    ('C06', 'Two shallow weighted skip junctions + HF', {'weighted_skip': True, 'hf': True}),
    ('C07', 'PPM/FPN UPerNet-derived decoder + bilinear', {'decoder': 'upernet'}),
    ('C08', 'Nested UNet++-derived decoder + bilinear', {'decoder': 'unetpp'}),
    ('C09', 'Parallel ResNet101/Swin-T + gated stage fusion', {'hybrid': 'gated'}),
    ('C10', 'Bottleneck four-direction Mamba (derived VSS-like) + HF', {'mamba_blocks': 1, 'hf': True}),
    ('C11', 'MicroNet v1.0 ResNet101 initialization + HF', {'initialization': 'micronet', 'hf': True}),
    ('C12', 'Multistage train-only SSL initialization + HF + cross', {'initialization': 'ssl', 'hf': True, 'cross': 'attention'}),
    ('CTRL_cross_concat', 'Layer3/layer4 concat instead of attention', {'cross': 'concat'}),
    ('CTRL_dilation_fixed', 'Fixed equal dilation branch weights', {'dilation': 'fixed'}),
    ('CTRL_weighted_only', 'Weighted skip without HF', {'weighted_skip': True}),
    ('CTRL_hybrid_concat', 'Parallel encoder concat fusion control', {'hybrid': 'concat'}),
    ('CTRL_mamba_only', 'Mamba without HF', {'mamba_blocks': 1}),
    ('CTRL_micronet_only', 'MicroNet initialization without HF', {'initialization': 'micronet'}),
    ('CTRL_ssl_only', 'Multistage SSL initialization without HF/cross', {'initialization': 'ssl'}),
]

if __name__ == '__main__':
    folder = Path(__file__).parent / 'configs'
    folder.mkdir(exist_ok=True)
    keep = {f'{experiment_id}.json' for experiment_id, _, _ in EXPERIMENTS}
    for path in folder.glob('*.json'):
        if path.name not in keep:
            path.unlink()
    for experiment_id, description, changes in EXPERIMENTS:
        config = new_config(experiment_id, description, **changes)
        (folder / f'{experiment_id}.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(EXPERIMENTS)} configurations; no training started.')
