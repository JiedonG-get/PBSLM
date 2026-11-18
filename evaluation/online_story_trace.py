"""Causal trace scorer for existing public story fixtures (no model execution).

Trace order is delivery/commit order, supplied by a trusted harness. Assertion
labels must come from an external annotator, not the backend under evaluation.
"""
import argparse
import json
import math
from pathlib import Path

from evaluation.online_story_contract import load_cases


def score_trace(case, trace):
    """Score annotated assertions at commit, never against future observations."""
    events = case['events']
    cursor = 0
    previous_time = -1
    target = None
    pending = None
    repairs = []
    before_end = stale = unverifiable = 0
    for row in trace:
        timestamp = row['ms']
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0 or timestamp < previous_time:
            raise ValueError('monotonic finite nonnegative trace time required')
        previous_time = timestamp
        if row['kind'] == 'input':
            if cursor == len(events) or row['event_id'] != events[cursor]['id']:
                raise ValueError('input order must match fixture delivery order')
            next_target = events[cursor]['target_after']
            if next_target != target:
                if pending is not None:
                    pending['status'] = 'superseded'
                pending = {'event_id': row['event_id'], 'input_ms': timestamp,
                           'status': 'unanswered', 'first_correct_after_ms': None}
                repairs.append(pending)
                target = next_target
            cursor += 1
        elif row['kind'] == 'commit':
            if cursor == 0 or not isinstance(row['text'], str) or not row['text']:
                raise ValueError('nonempty commit after first input required')
            assertion = row['assertion']
            if assertion is not None and (not isinstance(assertion, str) or not assertion):
                raise ValueError('assertion must be a nonempty label or null')
            if assertion is None:
                unverifiable += 1
                continue
            before_end += cursor < len(events)
            if assertion != target:
                stale += 1  # Includes incorrect assertions that were never true.
            elif pending is not None:
                pending['status'] = 'answered'
                pending['first_correct_after_ms'] = timestamp - pending['input_ms']
                pending = None
        else:
            raise ValueError('unknown trace event kind')
    if cursor != len(events):
        raise ValueError('incomplete fixture delivery')
    return {'incorrect_assertion_commits': stale, 'unscored_commits': unverifiable,
            'assertion_commits_before_last_input': before_end, 'target_intervals': repairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = {case['id']: case for case in load_cases()['cases']}
    traces = json.loads(args.traces.read_text(encoding='utf-8'))
    if not isinstance(traces, dict) or set(traces) != set(cases):
        raise ValueError('one trace per fixture case required')
    results = {name: score_trace(case, traces[name]) for name, case in cases.items()}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
