"""Generate matched 150-epoch ablations; generating configs never trains."""
import argparse
import copy
import json
from pathlib import Path

from komap_phase.config import BASE, validate


def configurations(seed=42, epochs=150):
    changes = {
        'B': ('R022 same-package baseline', False, False, 0., 1.),
        'D01': ('PSCL-inspired supervised multiscale local contrast; not full PSCL', False, True, .05, 1.),
        'D02': ('Derived BEM at half-resolution decoder feature; not full BE-UNet', True, False, 0., 1.),
        'D03': ('CE weights only at direct Al3Ni/Eutectic contacts', False, False, 0., 2.),
        'D04': ('Optional D01+D02 interaction; run only after separate ablations', True, True, .05, 1.),
    }
    result = []
    for name, (description, bem, contrast, weight, contact) in changes.items():
        c = copy.deepcopy(BASE)
        c.update(id=name, description=description)
        c['train'].update(seed=seed, epochs=epochs)
        c['model'].update(bem=bem, contrast=contrast)
        c['aux'].update(contrast_weight=weight, contact_weight=contact)
        result.append(validate(c))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--output', type=Path, default=Path('configs'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for c in configurations(a.seed, a.epochs):
        path = a.output/f'{c["id"]}.json'
        if path.exists() and json.loads(path.read_text()) != c:
            raise ValueError(f'Refuse to overwrite a different config: {path}')
        path.write_text(json.dumps(c, indent=2)+'\n')
    print(f'Generated 5 configs: seed={a.seed}, epochs={a.epochs}; no training started')
