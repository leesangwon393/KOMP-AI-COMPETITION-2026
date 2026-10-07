#!/usr/bin/env python3
"""Package source/config/docs/checks only; never package datasets or model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root.parents[1]/'releases/KoMaP_UNet_Transformer_Code_20261008.zip')
    args = parser.parse_args()
    files = list(root.glob('*.py')) + list(root.glob('*.sh')) + [root/n for n in ('README_KO.md','requirements.txt','requirements-cuda.txt')]
    for folder, pattern in [('komap_phase','*.py'),('configs','*.json'),('checks','cpu_*.json'),('docs','*')]:
        files += [p for p in (root/folder).glob(pattern) if p.is_file()]
    names = sorted({p.relative_to(root).as_posix() for p in files})
    manifest = dict(release=args.output.stem, implementation='implemented_cpu_verified',
        cuda_full_preflight='not_run', actual_transformer_training='not_started',
        data_included=False, required_dataset=dict(train=70,valid=20),
        files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names})
    (root/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in names + ['RELEASE_MANIFEST.json']:
            archive.write(root/name,Path(args.output.stem)/name)
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        assert not any('/Data/' in name or name.endswith(('.pt','.pth')) for name in archive.namelist())
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.sha256').write_text(digest+'  '+args.output.name+'\n')
    print(json.dumps(dict(zip=str(args.output),bytes=args.output.stat().st_size,sha256=digest,files=len(names)+1,data_included=False)))


if __name__=='__main__':
    main()
