"""Build a portable, checksummed ZIP. This script never starts training."""
import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path

from komap_lab.config import load_config

ROOT = Path(__file__).resolve().parent
IGNORED_DIRS = {'__pycache__', '.venv', '.git', 'runs', 'weights'}
IGNORED_FILES = {'.DS_Store', 'SHA256SUMS.txt'}


def files_to_ship():
    paths = []
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(p in IGNORED_DIRS for p in relative.parts):
            continue
        if path.name in IGNORED_FILES or path.suffix in ('.pyc', '.zip', '.pt', '.tmp'):
            continue
        if relative.parts[0] == 'Data' and path.suffix != '.png':
            continue
        paths.append(path)
    return paths


def build(output):
    for source in sorted(ROOT.rglob('*.py')):
        if any(part in IGNORED_DIRS for part in source.relative_to(ROOT).parts):
            continue
        ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    configs = {path.stem: load_config(path) for path in sorted((ROOT / 'configs').glob('*.json'))}
    required = {'B', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12'}
    if not required <= configs.keys():
        raise ValueError(f'Missing configs: {sorted(required - configs.keys())}')
    baseline = configs['B']
    if (baseline['train']['patch'], baseline['train']['target'], baseline['train']['epochs'],
            baseline['model']['upsampling'], baseline['loss']['class_weights']) != (448, 224, 150, 'bilinear', [1, 1, 2, 1]):
        raise ValueError('B does not match the documented R022 recipe')
    counts = {split: len(list((ROOT / 'Data' / split / 'images').glob('*.png')))
              for split in ('train', 'valid', 'test')}
    if counts != {'train': 70, 'valid': 20, 'test': 10}:
        raise ValueError(f'Unexpected image counts: {counts}')
    for split, expected in (('train', 70), ('valid', 20)):
        masks = len(list((ROOT / 'Data' / split / 'masks').glob('*.png')))
        if masks != expected:
            raise ValueError(f'Unexpected {split} masks: {masks}')
    paths = files_to_ship()
    digests = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    hashes = ROOT / 'SHA256SUMS.txt'
    hashes.write_text(''.join(f'{digest}  {name}\n' for name, digest in digests.items()), encoding='utf-8')
    paths.append(hashes)
    if output.exists():
        raise ValueError(f'ZIP exists; choose a new output name: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, f'{ROOT.name}/{path.relative_to(ROOT).as_posix()}')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(paths):
            raise ValueError('ZIP CRC or entry-count verification failed')
        for name, digest in digests.items():
            if hashlib.sha256(archive.read(f'{ROOT.name}/{name}')).hexdigest() != digest:
                raise ValueError(f'ZIP SHA256 mismatch: {name}')
    print(json.dumps({'zip': str(output.resolve()), 'bytes': output.stat().st_size,
                      'files': len(paths), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'configs': len(configs), 'dataset_counts': counts,
                      'training_started': False}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT.parent / 'KoMaP_R022_Context448_Architecture_Experiments_20261001.zip')
    build(parser.parse_args().output)
