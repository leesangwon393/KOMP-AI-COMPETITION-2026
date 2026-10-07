import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
manifest=json.loads((root/'release_manifest.json').read_text())
for name,entry in manifest['files'].items():
    path=root/name
    if not path.is_file() or path.stat().st_size!=entry['bytes']:
        raise RuntimeError('Release file/size mismatch: '+name)
    with path.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=entry['sha256']:
            raise RuntimeError('Release hash mismatch: '+name)
print(json.dumps({'release_verified':True,'files':len(manifest['files']),'origin':manifest['origin']}))
