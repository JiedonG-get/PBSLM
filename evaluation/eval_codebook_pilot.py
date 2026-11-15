"""Evaluate existing paired pilot files with the historical fixed held protocol."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / 'experiments/phase1_2/pilot1500'
    for tag in ('trainable', 'frozen'):
        target = directory / f'eval_{tag}.json'
        if target.exists():
            raise FileExistsError(target)
        command = [sys.executable, '-m', 'evaluation.phase1_1_eval', '--checkpoint',
                   str(directory / f'{tag}.pt'), '--config', str(directory / 'eval_config.json'), '--tag', tag]
        with (directory / f'eval_{tag}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f'{tag}: exit={result.returncode}', flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
