import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / 'configs/B.json').read_text())


def validate(c):
    if set(c) != set(BASE) or c['schema_version'] != 4:
        raise ValueError('Unsupported configuration schema')
    for key in ('model', 'normalization', 'train', 'loss', 'evaluation', 'aux'):
        if set(c[key]) != set(BASE[key]):
            raise ValueError(f'Missing/extra fields in {key}')
    m, t, a = c['model'], c['train'], c['aux']
    if m['backbone'] not in ('resnet101','resnet50','resnet34','convnext_tiny') or m['initialization'] not in ('imagenet','none'):
        raise ValueError('Unsupported backbone/initialization')
    if m['output_stride'] not in (16,32) or (m['backbone']!='resnet101' and m['output_stride']!=32):
        raise ValueError('OS16 is supported only for ResNet101')
    if any(type(m[k]) is not bool for k in ('bem', 'contrast')):
        raise ValueError('bem/contrast must be boolean')
    if m['rrcu'] not in ('none','feature','spatial_gate','channel_gate'):
        raise ValueError('Invalid RRCU placement')
    if m['bem'] or m['contrast'] or a['contrast_weight']:
        raise ValueError('Development trials disable BEM/contrast')
    for key in ('epochs', 'patch', 'target', 'stride', 'samples_per_epoch', 'batch_size'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f'Invalid train.{key}')
    if t['patch'] != 2*t['target'] or t['patch'] % 32 or t['target'] % 32:
        raise ValueError('Context must be twice center target, both multiples of 32')
    if not 0 < t['stride'] <= t['target'] or t['batch_size'] < 2 or t['samples_per_epoch'] % t['batch_size']:
        raise ValueError('Invalid stride/batch/draw count')
    if type(t['seed']) is not int or t['seed'] < 0 or type(t['deterministic']) is not bool:
        raise ValueError('Invalid seed/deterministic flag')
    if t['workers'] != 0 or t['precision'] != 'fp32' or t['rare_classes'] != [1, 2]:
        raise ValueError('Recipe requires workers0, FP32, rare classes1/2')
    for key, value in [('brightness_contrast_probability', .5), ('gamma_probability', .3), ('noise_probability', .15)]:
        if t[key] != value:
            raise ValueError('Photometric probabilities are fixed by the dataset implementation')
    if not 0 <= t['rare_fraction'] <= 1 or not 0 <= t['enriched_probability'] <= 1:
        raise ValueError('Invalid sampling fractions')
    if not math.isfinite(t['lr']) or t['lr'] <= 0 or not math.isfinite(t['weight_decay']) or t['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')
    if c['normalization'] != BASE['normalization']:
        raise ValueError('R022 loss and RGB normalization are fixed')
    if m['skip_fusion'] not in ('none','freq_nooffset') or m['upsampling'] not in ('bilinear','pixelshuffle') or type(m['connectivity']) is not bool:
        raise ValueError('Invalid follow-up architecture')
    if m['backbone']!='resnet101' or m['rrcu']!='none' or m['output_stride']!=32:
        raise ValueError('Follow-ups require the fixed ResNet101 OS32 baseline')
    weights=c['loss']
    if weights['ce_weight']!=.5 or weights['class_weights']!=[1,1,2,1] or not any(abs(weights['dice_weight']-d)<1e-8 and abs(weights['lovasz_weight']-l)<1e-8 for d,l in ((.25,.25),(.35,.15),(.15,.35))):
        raise ValueError('Unsupported loss recipe')
    if a['connectivity_weight'] not in (0.,.05) or bool(a['connectivity_weight'])!=m['connectivity']:
        raise ValueError('Connectivity head and loss must match')
    if c['evaluation']['absent'] != 'one' or c['evaluation']['batch_size'] < 1 or type(c['evaluation']['final_d4']) is not bool:
        raise ValueError('Invalid evaluation settings')
    for key in ('temperature', 'purity', 'contrast_weight', 'contact_weight'):
        if not isinstance(a[key], (int, float)) or not math.isfinite(a[key]):
            raise ValueError(f'Invalid aux.{key}')
    for key in ('samples_per_class', 'embedding_width', 'warmup_epochs', 'ramp_epochs', 'contact_radius'):
        if type(a[key]) is not int:
            raise ValueError(f'aux.{key} must be an integer')
    if a['temperature'] <= 0 or not .5 < a['purity'] <= 1 or a['contrast_weight'] < 0 or a['contact_weight'] < 1:
        raise ValueError('Invalid contrast/contact settings')
    if a['samples_per_class'] < 2 or a['embedding_width'] < 1 or a['warmup_epochs'] < 0 or a['ramp_epochs'] < 1 or a['contact_radius'] < 0:
        raise ValueError('Invalid contrast sampling/ramp/radius')
    if not m['contrast'] and a['contrast_weight'] != 0:
        raise ValueError('Contrast loss requires contrast heads')
    return c


def load_config(path):
    return validate(json.loads(Path(path).read_text()))


def protocol(c):
    c = copy.deepcopy(c)
    for key in ('id', 'description', 'aux'):
        c.pop(key)
    for key in ('bem', 'contrast', 'rrcu', 'backbone', 'output_stride','skip_fusion','upsampling','connectivity'):
        c['model'].pop(key)
    c['train'].pop('lr')
    for key in ('dice_weight','lovasz_weight'):
        c['loss'].pop(key)
    return c
