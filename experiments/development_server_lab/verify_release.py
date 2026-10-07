"""Verify all files listed in the release checksum manifest."""
import hashlib
from pathlib import Path


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    count=0
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        expected,name=line.split('  ',1)
        path=(root/name).resolve()
        if root not in path.parents or not path.is_file(): raise SystemExit(f'Missing/invalid release file: {name}')
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
        if digest.hexdigest()!=expected: raise SystemExit(f'Checksum mismatch: {name}')
        count+=1
    print(f'PASS: {count} release files match SHA256SUMS.txt')
