"""Hash existing artifacts without editing them; paths are workspace-relative."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def describe(relative):
    path = ROOT / relative
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return {'path': relative, 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('paths', nargs='+')
    args = p.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError(out)
    report = {'hash_algorithm': 'sha256', 'artifacts': [describe(name) for name in args.paths]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
