"""Record software/source provenance for future experiment runs, without credentials."""
import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def capture(paths):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    sources = {}
    for name in paths:
        data = (ROOT / name).read_bytes()
        sources[name] = hashlib.sha256(data).hexdigest()
    return {'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short'),
            'source_sha256_at_capture': sources, 'python': sys.version,
            'platform': platform.platform(), 'torch': torch.__version__,
            'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'limitation': 'dirty status is explicit; hashes identify exact selected sources; GPU bitwise determinism not promised'}
