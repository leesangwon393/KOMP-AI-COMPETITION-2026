"""Hybrid schema; validate the unchanged science recipe with the original validator."""
import copy
import json
from pathlib import Path
from .legacy_config import validate as validate_legacy

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / 'configs/B.json').read_text())


def validate(c):
    if set(c) != set(BASE) or c['schema_version'] != 5:
        raise ValueError('Expected hybrid configuration schema5')
    h = c['transformer']
    if set(h) != set(BASE['transformer']):
        raise ValueError('Missing/extra transformer fields')
    if h['variant'] not in ('B', 'HT_BOT', 'HT_PYR', 'HT_DEC16') or c['id'] != h['variant']:
        raise ValueError('id must match a supported transformer variant')
    if h != dict(BASE['transformer'], variant=c['id']):
        raise ValueError('Transformer optimization/initialization recipe is fixed')
    legacy = copy.deepcopy(c)
    legacy.pop('transformer')
    legacy['schema_version'] = 4
    if c['train']['batch_size'] not in (4, 8, 16):
        raise ValueError('Supported batch sizes are4/8/16; compare against same-batch B')
    legacy['train']['batch_size'] = 4
    validate_legacy(legacy)
    if c['model']['units'] or c['model']['backbone'] != 'resnet101':
        raise ValueError('Hybrid experiments require the unchanged R101-scSE base')
    return c


def load_config(path):
    return validate(json.loads(Path(path).read_text()))


def protocol(c):
    """Paired science recipe, excluding the architecture and its defined treatment."""
    c = copy.deepcopy(c)
    for key in ('id', 'description', 'transformer'):
        c.pop(key)
    return c
