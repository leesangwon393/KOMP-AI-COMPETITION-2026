"""Build a source+dataset ZIP; verify tests, CRC and every SHA256. Never train."""
import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path

from komap_phase.config import ROOT, load_config
from komap_phase.data import manifest
from komap_phase.runtime import code_hash
from server_suite import PROFILES

PACKAGE = 'KoMaP_U2Net_RSU_Server_20261004'


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_report(name):
    report = json.loads((ROOT / 'verification' / name).read_text())
    recorded = report.get('code_sha256', report.get('environment', {}).get('code_sha256'))
    if recorded != code_hash():
        raise ValueError(f'Verification used different executable code: {name}')
    return report


def build(data, output, test_output):
    if output.exists() or test_output.exists():
        raise ValueError('Choose new paths for both release ZIPs')
    configs = {path.stem: load_config(path) for path in sorted((ROOT / 'configs').glob('*.json'))}
    if len(configs) != 14 or any((c['train']['epochs'], c['train']['seed'], c['train']['patch'],
                                c['train']['target'], c['train']['batch_size']) != (150, 42, 448, 224, 4)
                               for c in configs.values()):
        raise ValueError('Release requires14 configs at150/seed42/context448/center224/batch4')
    sources = [p for p in sorted(ROOT.iterdir()) if p.is_file() and p.suffix in ('.py', '.md', '.txt')]
    sources.append(ROOT / '.gitignore')
    for folder in ('komap_phase', 'configs', 'tests', 'verification', 'docs'):
        sources.extend(p for p in sorted((ROOT / folder).rglob('*')) if p.is_file()
                       and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.md', '.txt'))
    for source in sources:
        if source.suffix == '.py':
            ast.parse(source.read_text(), filename=str(source))
    checked = read_report('synthetic_cpu.json')
    if len(checked['checks']) != 14 or {v['id'] for v in checked['checks']
                                      if v['status'] == 'passed' and len(v['steps']) == 2} != set(configs):
        raise ValueError('Missing successful two-step model checks')
    unit = read_report('unit_tests.json')
    reference = read_report('reference_baseline.json')
    smoke = read_report('server_plan_smoke.json')
    if not unit['passed'] or unit['tests'] != 13:
        raise ValueError('13 unit/integration tests must pass')
    if not reference['passed'] or not smoke['passed'] or not smoke['resume_completed_runs_skipped']:
        raise ValueError('Baseline identity and supervisor/resume checks must pass')
    for name in ('U2', 'F3'):
        report = read_report(f'fullsize_{name}.json')
        if len(report['checks']) != 1:
            raise ValueError(f'Invalid full-size report: {name}')
        entry = report['checks'][0]
        if (entry['id'], entry['status'], entry['synthetic_size'], entry['synthetic_batch_size']) != (name, 'passed', 448, 4):
            raise ValueError(f'Missing full-size448/batch4 check: {name}')
    provenance = json.loads((ROOT / 'docs/U2NET_SOURCE_PROVENANCE.json').read_text())
    for item in provenance['files'].values():
        if digest_file(ROOT / item['path']) != item['sha256']:
            raise ValueError('Vendored upstream snapshot/license changed')
    dataset_manifest = manifest(data, patch=224)
    counts = {}
    datasets = []
    test_images = []
    for split in ('train', 'valid', 'test'):
        images = sorted((data / split / 'images').glob('*.png'))
        masks = sorted((data / split / 'masks').glob('*.png')) if split != 'test' else []
        counts[split] = {'images': len(images), 'masks': len(masks)}
        if split == 'test':
            test_images = images
        else:
            datasets.extend(images + masks)
    if counts != {'train': {'images': 70, 'masks': 70}, 'valid': {'images': 20, 'masks': 20},
                  'test': {'images': 10, 'masks': 0}}:
        raise ValueError(f'Unexpected dataset counts: {counts}')
    test_entries = {'Data/' + p.relative_to(data).as_posix(): p for p in test_images}
    test_checksums = {name: digest_file(path) for name, path in test_entries.items()}
    test_output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(test_output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(test_entries.items()):
            archive.write(path, f'{PACKAGE}/{name}')
        sums = ''.join(f'{sha}  {name}\n' for name, sha in sorted(test_checksums.items()))
        archive.writestr(f'{PACKAGE}/TEST10_SHA256SUMS.txt', sums)
    with zipfile.ZipFile(test_output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Test10 ZIP CRC check failed')
        for name, expected in test_checksums.items():
            if hashlib.sha256(archive.read(f'{PACKAGE}/{name}')).hexdigest() != expected:
                raise ValueError(f'Test10 ZIP checksum mismatch: {name}')
    entries = {p.relative_to(ROOT).as_posix(): p for p in sources}
    entries.update({'Data/' + p.relative_to(data).as_posix(): p for p in datasets})
    checksums = {name: digest_file(path) for name, path in entries.items()}
    metadata = {
        'package': PACKAGE, 'date': '2026-10-04', 'configs': list(configs), 'dataset_counts': counts,
        'main_zip_splits': ['train', 'valid'],
        'optional_test_zip': {'name': test_output.name, 'bytes': test_output.stat().st_size,
                              'sha256': digest_file(test_output), 'images': 10,
                              'required_for_training_or_validation': False},
        'new_komap_training_started': False, 'trained_checkpoints_included': False,
        'default_plan': {'stage': 'core', 'ids': PROFILES['core'], 'seeds': [42], 'epochs': 150,
                         'batch_size': 4, 'runs': 5},
        'optional_stages': PROFILES, 'code_sha256': code_hash(),
        'verification': {'device': 'cpu', 'cuda_validated': False, 'synthetic_models': 14,
                         'unit_tests': unit['tests'], 'fullsize_context': 448, 'fullsize_batch': 4,
                         'fullsize_models': ['U2', 'F3'], 'baseline_identity': True, 'plan_resume': True},
        'train_valid_identity': dataset_manifest,
        'source_checksums': {k: v for k, v in checksums.items() if not k.startswith('Data/')},
    }
    manifest_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2) + '\n').encode()
    checksums['RELEASE_MANIFEST.json'] = hashlib.sha256(manifest_bytes).hexdigest()
    sum_bytes = ''.join(f'{sha}  {name}\n' for name, sha in sorted(checksums.items())).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(entries.items()):
            archive.write(path, f'{PACKAGE}/{name}')
        archive.writestr(f'{PACKAGE}/RELEASE_MANIFEST.json', manifest_bytes)
        archive.writestr(f'{PACKAGE}/SHA256SUMS.txt', sum_bytes)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC check failed')
        for name, expected in checksums.items():
            digest = hashlib.sha256()
            with archive.open(f'{PACKAGE}/{name}') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f'ZIP checksum mismatch: {name}')
    if max(output.stat().st_size, test_output.stat().st_size) > 104857600:
        raise ValueError('A release ZIP exceeds the Drive connector100MiB limit')
    print(json.dumps({'zip': str(output.resolve()), 'bytes': output.stat().st_size,
                      'files': len(entries) + 2, 'sha256': digest_file(output),
                      'optional_test_zip': metadata['optional_test_zip'],
                      'dataset_counts': counts, 'new_komap_training_started': False}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-output', type=Path, required=True)
    args = parser.parse_args()
    build(args.data.resolve(), args.output.resolve(), args.test_output.resolve())
