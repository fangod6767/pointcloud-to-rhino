"""Download the pinned Open3D-ML checkpoint and verify SHA-256."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    info = json.loads((ROOT / 'weights/provenance.json').read_text(encoding='utf-8'))
    target = ROOT / 'weights/randlanet_s3dis.pth'
    expected = info['checkpoint_sha256']
    if target.exists():
        if digest(target) == expected:
            print('Checkpoint already verified:', target)
            return
        raise RuntimeError('Existing checkpoint has an unexpected hash; move it aside before retrying.')
    temporary = target.with_suffix('.pth.part')
    print('Downloading pretrained checkpoint (~60 MB)...')
    with urllib.request.urlopen(info['checkpoint_url'], timeout=60) as response, temporary.open('wb') as f:
        for block in iter(lambda: response.read(1024 * 1024), b''):
            f.write(block)
    if digest(temporary) != expected:
        temporary.unlink()
        raise RuntimeError('Checkpoint SHA-256 mismatch.')
    temporary.replace(target)
    print('Verified:', target)

if __name__ == '__main__':
    main()
