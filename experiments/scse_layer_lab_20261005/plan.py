"""Pure planning functions; selections are persisted before dependent runs start."""
import copy
from komap_phase.config import BASE, validate

SEEDS = (42, 43, 44)


def config_for(ident, seed, units=(), profile='full'):
    c = copy.deepcopy(BASE)
    c['id'], c['description'] = ident, ident + ': controlled decoder gate ablation'
    c['train']['seed'] = seed
    c['model']['units'] = list(copy.deepcopy(units))
    if profile == 'smoke':
        c['model'].update(backbone='resnet34', initialization='none')
        c['train'].update(epochs=2, patch=64, target=32, stride=32, samples_per_epoch=4, batch_size=2)
    return validate(c)


def unit(stage, mode='channel_gate', steps=2, sharing='shared'):
    return dict(stage=stage, mode=mode, steps=steps, sharing=sharing)


def reproduction(profile='full'):
    return [config_for(ident, seed, [] if ident=='B' else [unit(3)], profile)
            for seed in SEEDS for ident in ('B','C3')]


def positions(profile='full'):
    return [config_for('C'+str(stage), 42, [unit(stage)], profile) for stage in (0,1,2,4)]


def select_positions(scores):
    """D4 mIoU descending, then stable ID for ties; B is not a position."""
    return sorted(scores, key=lambda ident: (-scores[ident], ident))[:2]


def mechanisms(stage, profile='full'):
    return [config_for(ident, 42, [u], profile) for ident, u in (
        ('M_T1', unit(stage, steps=1)),
        ('M_UNSHARED', unit(stage, sharing='unshared')),
        ('M_SPATIAL', unit(stage, mode='spatial_gate')),
        ('M_BOTH', unit(stage, mode='both_gate')))]


def combinations(top2, profile='full'):
    return [config_for('C_TOP2', 42, [unit(int(ident[1:])) for ident in top2], profile),
            config_for('C_ALL', 42, [unit(stage) for stage in range(5)], profile)]


def select_final(candidates):
    # Choose by seed42 exploration only; seed43/44 do not enter candidate ranking.
    return sorted(candidates, key=lambda r: (-r['d4'], r['config']['id']))[:2]


def spatial_candidates(stage, profile='full'):
    return [config_for(ident,42,[unit(stage,mode=mode)],profile)
            for method in ('cbam','coordinate','multiscale')
            for ident,mode in [('S_'+method.upper(),method+'_gate'),
                               ('CS_'+method.upper(),'channel_'+method+'_gate')]]
