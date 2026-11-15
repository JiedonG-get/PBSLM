"""Prequential synthetic association experiment on real masked PBSLMv2.

Diagnostic, not a language benchmark. Predict before reveal/update; no future-label memory.
Run from root: python -m evaluation.continual_associations --seeds 0 --base-steps 300 --events 128
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from models.model_v2 import PBSLMv2

ROOT = Path(__file__).resolve().parents[1]


def digest(model, excluded=()):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if any(name == prefix or (prefix.endswith('.') and name.startswith(prefix)) for prefix in excluded):
            continue
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def data(keys, new=False):
    # 16 disjoint keys, 16 labels. Context prefix denotes domain.
    # Shift changes only half the keys; the other half measures retention.
    prefix = 2 if new else 1
    x = torch.stack([torch.full_like(keys, prefix), keys + 4,
                     torch.full_like(keys, prefix), keys + 4], dim=1)
    target = keys + 24
    if new:
        target = (keys + 3) % 8 + 24
    return x, target


def predict(model, x):
    return model(x, 1, sample_noise=False)


def decode(model, z):
    return (z.float() @ model.code.float().t()).argmax(-1).squeeze(-1)


def loss(model, z, y):
    # Small synthetic vocabulary; exact cosine classification objective.
    return F.cross_entropy((z[:, 0].float() @ model.code.float().t()) / .1, y)


def evaluate(model, device):
    result = {}
    with torch.no_grad():
        for name, keys, shift in [('old', torch.arange(8, 16), False),
                                  ('new', torch.arange(8), True),
                                  ('old_changed', torch.arange(8), False)]:
            x, y = data(keys.to(device), shift)
            out = predict(model, x)
            result[name] = {side: float((decode(model, out['z_' + side]) == y).float().mean())
                            for side in ('a', 'c')}
    return result


def run(seed, args):
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = PBSLMv2(vocab=48, k=8, depth=1, d=32, heads=4, m_max=2,
                    act_x=.5, e_n=2, e_depth=1, refine=1, field_scale=0,
                    trainable_target=False, use_mem=True, use_flow=True,
                    use_sample_masks=True).to(device).eval()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.002)
    keys = torch.arange(16, device=device)
    x, y = data(keys)
    start = time.perf_counter()
    for step in range(args.base_steps):
        # Base supervision is offline and explicitly separate from online events.
        out = predict(model, x)
        objective = loss(model, out['z_a'], y) + loss(model, out['z_c'], y)
        opt.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
    base = evaluate(model, device)
    state = copy.deepcopy(model.state_dict())
    base_seconds = time.perf_counter() - start
    arms = {}
    # Same event order and initialization across arms. Label arrives only after prediction.
    gen = torch.Generator().manual_seed(seed + 901)
    stream = torch.randint(0, 8, (args.events,), generator=gen).tolist()
    for arm in ('none', 'a_only', 'a_replay', 'memory_only', 'a_and_memory', 'shuffled_memory'):
        model.load_state_dict(state)
        params = model.freeze_for_online()
        optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=0)
        protected_exclusions = tuple(name + '.' for name in ('dec_a', 'refine_proj', 'ln_a', 'ctx_a', 'out_a')) + ('q_emb', 'q_pos', 'mem.')
        frozen_hash = digest(model, protected_exclusions)
        with torch.no_grad():
            c_before = predict(model, x)['z_c'].clone()
        correct = []
        for index, key in enumerate(stream):
            event_x, revealed_y = data(torch.tensor([key], device=device), new=True)
            with torch.no_grad():
                before = predict(model, event_x)
                correct.append(int(decode(model, before['z_a']).item() == revealed_y.item()))
            # Feedback is now revealed. Online update can see this label, never before.
            if arm in ('a_only', 'a_replay', 'a_and_memory'):
                update_x, update_y = event_x, revealed_y
                if arm == 'a_replay':
                    # Previously observed base examples, not future stream labels.
                    replay_keys = torch.tensor([index % 16, (index + 7) % 16], device=device)
                    replay_x, replay_y = data(replay_keys)
                    update_x = torch.cat((event_x, replay_x))
                    update_y = torch.cat((revealed_y, replay_y))
                out = predict(model, update_x)
                objective = loss(model, out['z_a'], update_y)
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(params, 1)
                optimizer.step()
            if arm in ('memory_only', 'a_and_memory', 'shuffled_memory'):
                # Keys are captured at observation time; values from just-arrived feedback.
                with torch.no_grad():
                    value_y = revealed_y
                    if arm == 'shuffled_memory':
                        value_y = (revealed_y - 24 + 1) % 8 + 24
                    model.write_memory(before['e_feats'], model.target_embed(value_y[:, None]), lr=.05)
                    model.decay_memory(.995)
        with torch.no_grad():
            c_after = predict(model, x)['z_c']
        if not torch.equal(c_before, c_after):
            raise AssertionError(f'C changed in {arm}')
        if frozen_hash != digest(model, protected_exclusions):
            raise AssertionError(f'Protected parameters changed in {arm}')
        arms[arm] = {'optimizer_steps': args.events if arm in ('a_only', 'a_replay', 'a_and_memory') else 0,
                     'examples_per_update': 3 if arm == 'a_replay' else (1 if arm in ('a_only', 'a_and_memory') else 0),
                     'prequential_accuracy': sum(correct) / len(correct),
                     'first_quarter_accuracy': sum(correct[:max(1,len(correct)//4)]) / max(1,len(correct)//4),
                     'last_quarter_accuracy': sum(correct[-max(1,len(correct)//4):]) / max(1,len(correct)//4),
                     'after': evaluate(model, device), 'c_bitwise_stable': True,
                     'protected_hash_stable': True, 'memory_norms': model.memory_norms()}
    return {'seed': seed, 'base': base, 'base_seconds': base_seconds,
            'total_seconds': time.perf_counter() - start, 'arms': arms}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--base-steps', type=int, default=300)
    p.add_argument('--events', type=int, default=128)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='experiments/continual_associations/pilot.json')
    args = p.parse_args()
    report = {'task': 'synthetic_domain_conditioned_key_to_label', 'limitation': 'not natural-language learning or runtime concurrency',
              'config': vars(args), 'torch': torch.__version__, 'runs': []}
    for seed in args.seeds:
        result = run(seed, args)
        report['runs'].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
