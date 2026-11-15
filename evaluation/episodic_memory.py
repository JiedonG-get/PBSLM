"""Train the *read* path for episodic associations; eval is parameter-update free.

Each episode has a fresh random mapping. Support labels are observed before memory
writes. Query labels supervise offline training only; held episodes never update
weights. Full-vocabulary cosine classification is used on this tiny task.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from models.model_v2 import PBSLMv2

ROOT = Path(__file__).resolve().parents[1]


def hash_parameters(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        if name.startswith('mem.'):
            continue
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def episode(gen, support, device):
    keys = torch.randperm(16, generator=gen)[:support].to(device) + 4
    labels = torch.randperm(8, generator=gen)[:support].to(device) + 24
    # Same observed key at support and query time: associative recall, not inference.
    return keys[:, None].expand(-1, 4).clone(), labels


@torch.no_grad()
def fill_memory(model, x, labels, mode):
    for mem in model.mem:
        mem.zero_()
    if mode == 'zero':
        return
    features, _ = model.e_forward(model.token_emb(x))
    if mode == 'wrong_values':
        # Rotate observed values while keeping keys fixed; use >=2 support examples.
        labels = labels.roll(1)
    values = model.target_embed(labels[:, None])
    model.write_memory(features, values, lr=.05)


def predictions(model, x):
    return model(x, 1, sample_noise=False)


def accuracy(model, z, y):
    logits = z[:, 0].float() @ model.code.float().t()
    return (logits.argmax(-1) == y).sum().item()


@torch.no_grad()
def held_eval(model, count, support, seed, device):
    gen = torch.Generator().manual_seed(seed)
    before = hash_parameters(model)
    sums = {mode: {'a': 0, 'c': 0, 'matches_written_value': 0, 'targets': 0, 'memory_norm_sum': 0.}
            for mode in ('normal', 'zero', 'wrong_values')}
    for i in range(count):
        x, y = episode(gen, support, device)
        c_reference = None
        for mode, metrics in sums.items():
            fill_memory(model, x, y, mode)
            out = predictions(model, x)
            if c_reference is None:
                c_reference = out['z_c'].clone()
            elif not torch.equal(c_reference, out['z_c']):
                raise AssertionError('C changed with memory intervention')
            for side in ('a', 'c'):
                metrics[side] += accuracy(model, out['z_' + side], y)
            written_y = y.roll(1) if mode == 'wrong_values' else y
            metrics['matches_written_value'] += accuracy(model, out['z_a'], written_y)
            metrics['targets'] += len(y)
            metrics['memory_norm_sum'] += sum(model.memory_norms())
    if before != hash_parameters(model):
        raise AssertionError('Evaluation changed parameters')
    for metrics in sums.values():
        metrics['a_accuracy'] = metrics['a'] / metrics['targets']
        metrics['c_accuracy'] = metrics['c'] / metrics['targets']
        metrics['written_value_match_rate'] = metrics['matches_written_value'] / metrics['targets']
        metrics['mean_memory_norm'] = metrics.pop('memory_norm_sum') / count
    return {'episodes': count, 'support_per_episode': support,
            'parameter_hash_unchanged': True, 'c_bitwise_stable': True,
            'arms': sums}


def run(seed, args):
    torch.manual_seed(seed)
    device = torch.device(args.device)
    m = PBSLMv2(vocab=48, k=8, depth=1, d=32, heads=4, m_max=2,
                act_x=.5, e_n=2, refine=1, field_scale=0,
                trainable_target=False, use_mem=True, use_flow=True,
                use_sample_masks=True).to(device).eval()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.002)
    train_gen = torch.Generator().manual_seed(seed + 1100)
    before = held_eval(m, 16, args.support, seed + 2200, device)
    start = time.perf_counter()
    curve = []
    for step in range(args.episodes):
        x, y = episode(train_gen, args.support, device)
        fill_memory(m, x, y, 'normal')
        out = predictions(m, x)
        logits = (out['z_a'][:, 0].float() @ m.code.float().t()) / .1
        objective = F.cross_entropy(logits, y)
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite episodic objective')
        opt.zero_grad(set_to_none=True)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if step == 0 or (step + 1) % 100 == 0:
            curve.append({'step': step + 1, 'loss': float(objective.detach()),
                          'grad_norm': float(norm),
                          'training_episode_accuracy': accuracy(m, out['z_a'], y) / len(y)})
    train_seconds = time.perf_counter() - start
    # New episode seed separate from before-eval and training.
    after = held_eval(m, args.eval_episodes, args.support, seed + 3300, device)
    capacity = {str(size): held_eval(m, args.eval_episodes, size, seed + 4400 + size, device)
                for size in (2, 8)}
    return {'seed': seed, 'before': before, 'after': after, 'capacity': capacity, 'curve': curve,
            'train_seconds': train_seconds, 'total_seconds': time.perf_counter() - start,
            'final_parameter_hash': hash_parameters(m)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--eval-episodes', type=int, default=64)
    p.add_argument('--support', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='experiments/episodic_memory/pilot.json')
    args = p.parse_args()
    if not 2 <= args.support <= 8:
        p.error('support must be 2..8 for nontrivial wrong-value controls')
    report = {'task': 'fresh_mapping_associative_recall', 'config': vars(args),
              'torch': torch.__version__, 'notes': ['offline read-path learning; evaluation only writes memory',
              'no query label used for eval parameter update', 'not language or runtime benchmark'], 'runs': []}
    for seed in args.seeds:
        result = run(seed, args)
        report['runs'].append(result)
        print(json.dumps(result), flush=True)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
