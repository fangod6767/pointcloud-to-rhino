"""Acquire one real room, run inference and prepare Notebook sampling paths."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    subprocess.run([sys.executable, str(ROOT/'download_weights.py')], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT/'prepare_sample.py'), '--room', 'Area_3/office_2'], check=True, cwd=ROOT)
    source = ROOT/'data/processed/Area_3_office_2.npz'
    target = ROOT/'data/full_processed/Area_3_office_2.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        import hashlib
        if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
            raise RuntimeError('Existing room differs; move it aside before retrying.')
    else:
        shutil.copy2(source, target)
    output = ROOT/'results/area3_batch/Area_3_office_2'
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Output already exists; inspect it before choosing whether to rerun inference.')
    subprocess.run([sys.executable, str(ROOT/'segmentation_pipeline.py'), str(target),
                    '--output', str(output), '--device', 'cpu', '--max-passes', '300', '--evaluate'], check=True, cwd=ROOT)
    print('Notebook sampling is ready for office_2. Results belong to this run, not the historical batch.')

if __name__ == '__main__':
    main()
