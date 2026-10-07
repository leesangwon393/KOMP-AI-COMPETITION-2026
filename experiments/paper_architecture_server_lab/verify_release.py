"""Verify all packaged files against SHA256SUMS; read-only."""
import hashlib
from pathlib import Path

def verify(root):
    count=0
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        expected,name=line.split('  ',1);path=root/name
        if not path.is_file():raise ValueError('Missing file: '+name)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=expected:raise ValueError('Changed file: '+name)
        count+=1
    print('Verified SHA256 for',count,'packaged files')

if __name__=='__main__':verify(Path(__file__).resolve().parent)
