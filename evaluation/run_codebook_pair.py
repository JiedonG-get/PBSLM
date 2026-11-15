"""Same-initialization codebook pilot using unchanged Phase1 trainer.

Fresh output directory required. This is a short pilot, not formal 15k-step history.
Paired models, data generator and sampled negatives use identical seeds/budgets.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from training.train_v21 import build_model, set_seed

ROOT = Path(__file__).resolve().parents[1]


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--output-dir', default='experiments/phase1_2/pilot1500')
    args = p.parse_args()
    out = ROOT / args.output_dir
    if out.exists():
        raise FileExistsError('Fresh directory required; do not append or overwrite research runs')
    out.mkdir(parents=True)
    configs = {}
    models = {}
    for tag in ('trainable', 'frozen'):
        cfg = json.loads((ROOT / f'configs/ctrl_codebook_{tag}.json').read_text(encoding='utf-8'))
        cfg['training']['steps'] = args.steps
        cfg['checkpoint']['save_every'] = 0
        cfg['checkpoint']['save'] = f'{args.output_dir}/{tag}.pt'
        cfg['logging']['metrics_jsonl'] = f'{args.output_dir}/{tag}_metrics.jsonl'
        cfg['logging']['eval_every'] = 500
        cfg['logging']['log_every'] = 100
        (out / f'{tag}_config.json').write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
        configs[tag] = cfg
        set_seed(cfg['seed'])
        models[tag] = build_model(cfg)
    left, right = models['trainable'], models['frozen']
    if set(left.state_dict()) != set(right.state_dict()):
        raise AssertionError('Initial state keys differ')
    equal = all(torch.equal(v, right.state_dict()[k]) for k, v in left.state_dict().items())
    if not equal:
        raise AssertionError('Initialization differs beyond requires_grad')
    evidence = {'initial_values_equal': equal, 'state_sha256': state_hash(left),
                'torch': torch.__version__, 'steps': args.steps, 'pilot_not_formal': True,
                'models': {tag: {'total_parameters': sum(p.numel() for p in m.parameters()),
                                 'trainable_parameters': sum(p.numel() for p in m.parameters() if p.requires_grad)}
                           for tag, m in models.items()}, 'runs': {}}
    (out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    del models, left, right
    for tag in ('trainable', 'frozen'):
        command = [sys.executable, '-m', 'training.train_v21', '--config',
                   str(out / f'{tag}_config.json')]
        start = time.perf_counter()
        with (out / f'{tag}.log').open('w', encoding='utf-8') as log:
            run = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        evidence['runs'][tag] = {'command': command, 'returncode': run.returncode,
                                 'seconds': time.perf_counter() - start}
        (out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(evidence['runs'][tag]), flush=True)
        if run.returncode:
            raise SystemExit(run.returncode)
    # Model files remain ignored/local; config/logs/metrics/evidence are publishable.


if __name__ == '__main__':
    main()
