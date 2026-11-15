"""Read-only local checkpoint provenance and compatibility evidence; never runs forward.

Run: python -m evaluation.check_checkpoint_compat --output experiments/o1_5/checkpoints.json
Only load locally trusted checkpoints. Artifacts remain local and ignored by Git.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from models.model_v2 import PBSLMv2

ROOT = Path(__file__).resolve().parents[1]


def inspect(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    args = ck.get('args', {})
    state = ck['state']
    cfg = dict(vocab=state['tok.weight'].shape[0], k=args['k'], depth=args['depth'],
               d=args['d'], heads=args['heads'], m_max=state['q_emb'].shape[0],
               act_x=args.get('act', .2), e_n=state['e_mix'].shape[0], e_depth=1,
               field_scale=args.get('field_scale', 0), use_mem=not args.get('no_mem', False),
               trainable_target=not args.get('frozen_code', False),
               simple_dec=args.get('simple_dec', False), dense=args.get('dense', False),
               direct_only=args.get('direct_only', False), use_flow=not args.get('no_flow', False))
    model = PBSLMv2(**cfg)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatches = {k: {'checkpoint': list(state[k].shape), 'model': list(expected[k].shape)}
                  for k in expected.keys() & state.keys() if state[k].shape != expected[k].shape}
    attempts = {}
    for mode, loader in [('raw', lambda: torch.nn.Module.load_state_dict(model, state, strict=True)),
                         ('out_c_migration', lambda: model.load_state_dict(state, strict=True))]:
        try:
            result = loader()
            attempts[mode] = {'success': True, 'missing': result.missing_keys,
                              'unexpected': result.unexpected_keys}
        except RuntimeError as exc:
            attempts[mode] = {'success': False, 'error': str(exc)}
    return {'path': str(path.resolve()), 'storage': 'local ignored artifact, not uploaded',
            'sha256': digest.hexdigest(), 'bytes': path.stat().st_size,
            'mtime_utc_not_verified_creation_time': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            'args': args, 'construction': cfg, 'inferred_defaults': ['e_depth=1', 'missing args use constructor-era defaults; not provenance proof'],
            'raw_missing': missing, 'raw_unexpected': unexpected, 'shape_mismatches': mismatches,
            'load_attempts': attempts, 'forward_executed': False,
            'limitation': 'loading is not functional equivalence; dense/direct_only/no_flow checkpoint does not validate sparse Flow'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='experiments/o1_5/checkpoints.json')
    ns = parser.parse_args()
    report = {'torch': torch.__version__, 'checkpoints': [inspect(ROOT / name) for name in
              ('ckpt_direct_ce.pt', 'ckpt_nomem_100.pt') if (ROOT / name).exists()]}
    out = ROOT / ns.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
