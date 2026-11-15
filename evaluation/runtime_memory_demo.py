"""Learned associative read + real runtime feedback/confirmation/restore demonstration.

Not a natural-language system. New small synthetic offline training is explicit;
all live session operations only write memory after feedback confirmation.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from evaluation.episodic_memory import run
from evaluation.provenance import capture
from runtime.model_session import CausalModelSession, state_hash
from runtime.session import StaleProposalError

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default='experiments/runtime_integration/demo.json')
    args = p.parse_args()
    path = ROOT / args.output
    if path.exists():
        raise FileExistsError(path)
    provenance = capture(['evaluation/runtime_memory_demo.py', 'evaluation/episodic_memory.py',
                          'runtime/model_session.py', 'runtime/session.py',
                          'models/model.py', 'models/model_v2.py'])
    model, training = run(args.seed, SimpleNamespace(device='cuda' if torch.cuda.is_available() else 'cpu',
                            support=4, episodes=500, eval_episodes=8), return_model=True)
    with torch.no_grad():
        for memory in model.mem:
            memory.zero_()
    session = CausalModelSession(model, 'learned-memory-demo')
    unchanged = state_hash(model)
    labels = [26, 24, 27, 25]
    records = []
    old_proposal = None
    for index, (key, target) in enumerate(zip(range(4, 8), labels)):
        observation = session.observe([key] * 4, event_time=index * 2 + 1)
        before = session.advance()
        proposal = session.propose(before['tokens_a'])
        encoded = json.dumps(before['tokens_a'])
        # First candidate is interrupted after one committed character.
        if index == 0:
            session.commit(proposal, expected_offset=0, count=1)
            session.interrupt('feedback_arriving')
            session.resume()
            session.advance()
            old_proposal = proposal
        else:
            session.commit(proposal, expected_offset=0, count=len(encoded))
        memory_before_feedback = state_hash(model, memory=True)
        candidate = session.stage_feedback(observation, [target], event_time=index * 2 + 2)
        if memory_before_feedback != state_hash(model, memory=True):
            raise AssertionError('feedback staging wrote M')
        receipt = session.confirm_memory(candidate)
        replay = session.confirm_memory(candidate)
        if not replay['replayed']:
            raise AssertionError('confirmation retry was not idempotent')
        records.append({'key': key, 'prediction_before_feedback': before['tokens_a'],
                        'revealed_feedback': target, 'memory_version': receipt['version']})
    stale_rejected = False
    try:
        session.commit(old_proposal, expected_offset=1, count=1)
    except StaleProposalError:
        stale_rejected = True
    snapshot = json.loads(json.dumps(session.snapshot()))
    restored = CausalModelSession.restore(model, snapshot)
    recall = []
    for index, (key, target) in enumerate(zip(range(4, 8), labels)):
        restored.observe([key] * 4, event_time=20 + index)
        prediction = restored.advance()
        recall.append({'key': key, 'target': target, 'tokens_a': prediction['tokens_a'],
                       'tokens_c': prediction['tokens_c'], 'correct': prediction['tokens_a'] == [target]})
    if unchanged != state_hash(model):
        raise AssertionError('live session changed model parameters or code buffer')
    report = {'scope': 'synchronous token-ID associative demonstration, not language/runtime concurrency',
              'seed': args.seed, 'provenance': provenance, 'offline_training': training,
              'feedback_trace': records, 'recall': recall,
              'recall_accuracy': sum(x['correct'] for x in recall) / len(recall),
              'stale_commit_rejected': stale_rejected,
              'live_parameter_hash_unchanged': True,
              'confirmed_versions': len(session.applied),
              'protocol_events': restored.protocol.events,
              'snapshot_protocol_and_numeric_memory_restored': True,
              'warning': 'support labels were explicitly observed; no factual correctness or general Online Learning claim'}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('recall_accuracy', 'stale_commit_rejected',
                        'live_parameter_hash_unchanged', 'confirmed_versions')}, indent=2))


if __name__ == '__main__':
    main()
