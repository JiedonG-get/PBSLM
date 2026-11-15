"""Recompute immediate corrective-feedback transitions from immutable event JSON."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='experiments/persistent_memory/three_seed.json')
    p.add_argument('--output', default='experiments/persistent_memory/correction_transitions.json')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    source = json.loads(Path(args.input).read_text(encoding='utf-8'))
    sums = {}
    for run in source['runs']:
        for arm, result in run['arms'].items():
            totals = sums.setdefault(arm, {'n': 0, 'already_correct_before': 0,
                                         'wrong_to_correct': 0, 'wrong_before': 0, 'correct_after': 0})
            for event in result['events']:
                if not event['is_correction']:
                    continue
                totals['n'] += 1
                before = event['prediction_before_feedback'] == event['truth']
                after = event['query_after_feedback'] == event['truth']
                totals['already_correct_before'] += int(before)
                totals['wrong_before'] += int(not before)
                totals['wrong_to_correct'] += int(not before and after)
                totals['correct_after'] += int(after)
    report = {'input': args.input, 'definition': 'immediate truth-relative transition, not delayed correction or verified uptake of wrong feedback',
              'arms': sums}
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
