"""Run evidence suite and retain raw stdout/stderr and environment with real exit code."""
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/o1_5/gpt_verification'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    modules = ['tests.test_flow_mask_min', 'tests.test_model_v2_mask_integration',
               'tests.test_online_o1', 'tests.test_masked_e2e_evidence',
               'tests.test_masked_boundaries']
    command = [sys.executable, '-m', 'unittest', *modules, '-v']
    run = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         encoding='utf-8', errors='replace')
    (OUT / 'unittest.log').write_text(run.stdout, encoding='utf-8')
    metadata = {'command': command, 'python': sys.version, 'torch': torch.__version__,
                'platform': platform.platform(), 'test_device': 'cpu',
                'returncode': run.returncode, 'training_run': False,
                'diagnostic_autograd': True, 'optimizer_step': False}
    (OUT / 'environment.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(run.stdout)
    raise SystemExit(run.returncode)


if __name__ == '__main__':
    main()
