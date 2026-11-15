"""Persistent associative-memory interference and explicit correction controls.

Offline episodic training is separate. During streaming eval only M changes;
queries are predicted before feedback arrives. Rebuild arms use an external latest-
feedback ledger and O(capacity) writes: NOT an intrinsic capability of matrix M.
"""
import argparse
from collections import OrderedDict
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from evaluation.episodic_memory import run, hash_parameters
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]


def context(keys, device):
    return torch.tensor(keys, device=device, dtype=torch.long)[:, None].expand(-1, 4).clone()


@torch.no_grad()
def recall(model, keys, device):
    out = model(context(keys, device), 1, sample_noise=False)
    return model.decode_nearest(out['z_a']).argmax(-1)[:, 0].tolist(), out['z_c'].clone()


@torch.no_grad()
def write(model, key, value, device):
    x = context([key], device)
    features, _ = model.e_forward(model.token_emb(x))
    values = model.target_embed(torch.tensor([[value]], device=device))
    model.write_memory(features, values, lr=.05)


def stream(seed, count):
    gen = torch.Generator().manual_seed(seed + 8000)
    events = []
    for i in range(count):
        key = int(torch.randint(4, 20, (), generator=gen))
        truth = int(torch.randint(24, 32, (), generator=gen))
        corrupt = i % 16 == 0
        events.append({'key': key, 'truth': truth,
                       'feedback': 24 + ((truth - 24 + 1) % 8) if corrupt else truth,
                       'is_corrupt': corrupt, 'is_correction': False})
        if corrupt:
            events.append({'key': key, 'truth': truth, 'feedback': truth,
                           'is_corrupt': False, 'is_correction': True})
    return events


@torch.no_grad()
def evaluate_arm(model, events, arm, device):
    for mem in model.mem:
        mem.zero_()
    immutable = hash_parameters(model)
    truth, ledger, last_seen = {}, OrderedDict(), {}
    traces = []
    exact_before = 0
    repeated = 0
    remembered = {'recent_0_3': [0, 0], 'age_4_15': [0, 0], 'age_16_plus': [0, 0]}
    write_calls = 0
    fixed_input = context(list(range(4, 20)), device)
    c_base = model(fixed_input, 1, sample_noise=False)['z_c'].clone()
    for index, event in enumerate(events):
        key, answer = event['key'], event['truth']
        before, _ = recall(model, [key], device)
        prior_truth = truth.get(key)
        if prior_truth is not None:
            exact_before += int(before[0] == prior_truth)
            repeated += 1
        # New hidden truth is recorded by evaluator only. Model sees feedback afterward.
        truth[key] = answer
        last_seen[key] = index
        if arm == 'append':
            write(model, key, event['feedback'], device)
            write_calls += 1
        elif arm == 'decay_append':
            model.decay_memory(.95)
            write(model, key, event['feedback'], device)
            write_calls += 1
        elif arm.startswith('rebuild'):
            capacity = int(arm.removeprefix('rebuild'))
            ledger.pop(key, None)
            ledger[key] = event['feedback']
            while len(ledger) > capacity:
                ledger.popitem(last=False)
            for mem in model.mem:
                mem.zero_()
            for item_key, item_value in ledger.items():
                write(model, item_key, item_value, device)
                write_calls += 1
        elif arm != 'zero':
            raise ValueError(arm)
        keys = sorted(truth)
        predictions, _ = recall(model, keys, device)
        hits = [int(pred == truth[k]) for k, pred in zip(keys, predictions)]
        for k, hit in zip(keys, hits):
            age = index - last_seen[k]
            bucket = 'recent_0_3' if age <= 3 else ('age_4_15' if age <= 15 else 'age_16_plus')
            remembered[bucket][0] += hit
            remembered[bucket][1] += 1
        observed_query, _ = recall(model, [key], device)
        traces.append({**event, 'event': index, 'prediction_before_feedback': before[0],
                       'prior_truth': prior_truth, 'query_after_feedback': observed_query[0],
                       'latest_truth_accuracy': sum(hits) / len(hits),
                       'ledger_items': len(ledger), 'mean_memory_norm': sum(model.memory_norms()) / model.e_n})
        c_now = model(fixed_input, 1, sample_noise=False)['z_c']
        if not torch.equal(c_now, c_base):
            raise AssertionError('C changed during memory-only stream')
    if hash_parameters(model) != immutable:
        raise AssertionError('Streaming evaluation changed model parameters')
    corrections = [item for item in traces if item['is_correction']]
    return {'events': traces, 'write_calls': write_calls,
            'average_latest_truth_accuracy': sum(t['latest_truth_accuracy'] for t in traces) / len(traces),
            'final_latest_truth_accuracy': traces[-1]['latest_truth_accuracy'],
            'prior_truth_accuracy_on_revisits': exact_before / repeated if repeated else None,
            'immediate_correct_after_correction': sum(t['query_after_feedback'] == t['truth'] for t in corrections) / len(corrections),
            'age_buckets': {k: {'correct': v[0], 'n': v[1], 'accuracy': v[0] / v[1] if v[1] else None}
                            for k, v in remembered.items()},
            'c_bitwise_stable': True, 'parameters_unchanged': True,
            'external_ledger': arm.startswith('rebuild')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--events', type=int, default=96)
    p.add_argument('--training-episodes', type=int, default=500)
    p.add_argument('--output', default='experiments/persistent_memory/three_seed.json')
    args = p.parse_args()
    path = ROOT / args.output
    if path.exists():
        raise FileExistsError(path)
    if args.events < 16 or args.training_episodes < 1:
        p.error('need >=16 events and positive training episodes')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report = {'config': vars(args), 'device': device,
              'provenance': capture(['evaluation/persistent_memory.py', 'evaluation/episodic_memory.py',
                                      'models/model.py', 'models/model_v2.py']),
              'limitations': ['synthetic familiar vocab', 'rebuild has external oracle-like latest feedback storage',
                              'hidden truth only evaluator; wrong feedback deliberately conflicts',
                              'not equal compute; write_calls reported'], 'runs': []}
    for seed in args.seeds:
        model, base = run(seed, SimpleNamespace(device=device, support=4,
                           episodes=args.training_episodes, eval_episodes=8), return_model=True)
        sequence = stream(seed, args.events)
        arms = {}
        for arm in ('zero', 'append', 'decay_append', 'rebuild8', 'rebuild16'):
            start = time.perf_counter()
            arms[arm] = evaluate_arm(model, sequence, arm, device)
            arms[arm]['seconds'] = time.perf_counter() - start
        report['runs'].append({'seed': seed, 'base_read_training': base, 'arms': arms})
        print(json.dumps({'seed': seed, 'arms': {a: {k: v for k, v in r.items() if k != 'events'}
                                                for a, r in arms.items()}}), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
