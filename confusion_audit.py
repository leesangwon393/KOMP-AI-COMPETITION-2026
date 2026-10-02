"""Audit saved D4 masks without training or modifying previous experiments."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/komap-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from mimu.data import decode_mask, CLASS_NAMES
from mimu.metrics import confusion_matrix, summarize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/confusion_audit_v1'
FOLDERS = {
    'baseline': ROOT/'runs/resnet34_unet_scse_ce_dice/d4_tta',
    'cldice': ROOT/'runs/research_sweep_v1/cldice/d4_tta',
    'dysample': ROOT/'runs/upsampling_sweep_v1/dysample/d4_tta',
}


def read(path):
    with Image.open(path) as im:
        return decode_mask(np.asarray(im.convert('RGB')))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT/'Data/valid/masks').glob('*.png'))
    assert len(paths) == 20
    matrices = {m: [] for m in FOLDERS}
    regions = {m: {k: np.zeros((4, 4), dtype=np.int64) for k in ('boundary', 'interior')} for m in FOLDERS}
    rows, candidates = [], []
    for path in paths:
        gt = read(path)
        boundary = ndi.maximum_filter(gt, size=3) != ndi.minimum_filter(gt, size=3)
        boundary = ndi.binary_dilation(boundary, iterations=1)
        for method, folder in FOLDERS.items():
            pr = read(folder/path.name)
            cm = confusion_matrix(gt, pr)
            matrices[method].append(cm)
            for key, mask in [('boundary', boundary), ('interior', ~boundary)]:
                regions[method][key] += confusion_matrix(gt[mask], pr[mask])
            rows.append({'file': path.name, 'method': method, 'miou': summarize([cm])['miou'],
                         'eut_to_al': int(cm[2, 3]), 'al_to_eut': int(cm[3, 2]),
                         'al3ni_to_eut': int(cm[1, 2]), 'eut_to_al3ni': int(cm[2, 1])})
            if method == 'baseline':
                for y in range(0, gt.shape[0]-127, 128):
                    for x in range(0, gt.shape[1]-127, 128):
                        g, p = gt[y:y+128, x:x+128], pr[y:y+128, x:x+128]
                        for a, b in [(2, 3), (3, 2), (1, 2), (2, 1)]:
                            candidates.append({'file': path.name, 'x': x, 'y': y, 'true': a, 'pred': b,
                                               'score': int(((g == a) & (p == b)).sum())})
    result = {'classes': CLASS_NAMES, 'orientation': 'rows=ground truth; columns=prediction',
              'boundary_definition': '3x3 GT label transition, dilated once (cross connectivity)',
              'models': {}}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, (method, cms) in zip(axes, matrices.items()):
        cm = sum(cms)
        scores = summarize(cms)
        saved = json.loads((FOLDERS[method]/'metrics.json').read_text())
        assert abs(scores['miou']-saved['miou']) < 1e-10, method
        recall = cm / cm.sum(1, keepdims=True)
        precision = np.diag(cm) / cm.sum(0)
        image_recall = np.array([np.divide(c, c.sum(1, keepdims=True),
                                out=np.full((4, 4), np.nan), where=c.sum(1, keepdims=True)>0) for c in cms])
        result['models'][method] = {'scores': scores, 'counts': cm.tolist(),
            'row_percent': (recall*100).tolist(), 'image_macro_row_percent': (np.nanmean(image_recall, axis=0)*100).tolist(),
            'precision': precision.tolist(), 'recall': np.diag(recall).tolist(),
            'regions': {k: v.tolist() for k, v in regions[method].items()}}
        im = ax.imshow(recall*100, vmin=0, vmax=100, cmap='Blues')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{recall[i,j]*100:.2f}%', ha='center', va='center',
                        color='white' if recall[i,j] > .55 else 'black')
        ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               xlabel='Prediction', ylabel='Ground truth', title=method)
        ax.tick_params(axis='x', labelrotation=30)
    fig.colorbar(im, ax=axes, label='Percent of ground-truth class')
    fig.savefig(OUT/'confusion_matrices.png', dpi=180)
    plt.close(fig)
    selected = []
    for a, b in [(2, 3), (3, 2), (1, 2), (2, 1)]:
        used = set()
        for row in sorted((r for r in candidates if (r['true'], r['pred']) == (a, b)), key=lambda r: -r['score']):
            if row['file'] in used:
                continue
            selected.append(row)
            used.add(row['file'])
            if len(used) == 2:
                break
    canvas = Image.new('RGB', (1280, 30+len(selected)*290), 'white')
    draw = ImageDraw.Draw(canvas)
    for j, title in enumerate(['Input', 'Ground truth', *FOLDERS]):
        draw.text((j*256+8, 8), title, fill='black')
    for i, row in enumerate(selected):
        files = [ROOT/'Data/valid/images'/row['file'].replace('_mask', '_image'),
                 ROOT/'Data/valid/masks'/row['file'], *[d/row['file'] for d in FOLDERS.values()]]
        box = (row['x'], row['y'], row['x']+128, row['y']+128)
        for j, file in enumerate(files):
            with Image.open(file) as pic:
                canvas.paste(pic.convert('RGB').crop(box).resize((256, 256), Image.Resampling.NEAREST), (j*256, 30+i*290))
        draw.text((8, 288+i*290), f"{CLASS_NAMES[row['true']]} -> {CLASS_NAMES[row['pred']]} | {row['file']} | x={row['x']},y={row['y']} | baseline pixels={row['score']}", fill='black')
    canvas.save(OUT/'error_panels.png')
    result['selected_crops'] = selected
    result['crop_selection'] = 'Top baseline directional errors in 128px grid; two distinct images per direction. Descriptive, not representative.'
    (OUT/'audit.json').write_text(json.dumps(result, indent=2))
    with (OUT/'per_image.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result['models'], indent=2))


if __name__ == '__main__':
    main()
