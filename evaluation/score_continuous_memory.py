"""Score sealed predictions for continuous-memory queries against held-out labels."""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import unicodedata

from evaluation.continuous_event_protocol import load_streams


UNKNOWN = '\u4e0d\u786e\u5b9a'
TERMINAL_PUNCTUATION = '.!?,;:\u3002\u3001'


def normalize_answer(answer):
    """Normalize width and remove surrounding whitespace and sentence punctuation."""
    answer = unicodedata.normalize('NFKC', answer).strip()
    while answer and answer[-1] in TERMINAL_PUNCTUATION:
        answer = answer.rstrip(TERMINAL_PUNCTUATION).rstrip()
    return answer


def query_gold(streams):
    gold = {}
    for stream in streams:
        for event in stream['events']:
            if event['kind'] != 'query':
                continue
            key = stream['stream_id'], event['event_id']
            oracle = event.get('oracle')
            if not isinstance(oracle, dict) or not isinstance(oracle.get('target'), str):
                raise ValueError(f'missing query oracle target for {key!r}')
            case = oracle.get('case', 'unspecified')
            target = normalize_answer(oracle['target'])
            if not target or not isinstance(case, str) or not case.strip():
                raise ValueError(f'invalid query oracle target or case for {key!r}')
            gold[key] = {'case': case, 'target': target, 'known': target != UNKNOWN}
    if not gold:
        raise ValueError('at least one fixture query required')
    return gold


def index_predictions(predictions, gold):
    if not isinstance(predictions, dict):
        raise ValueError('prediction document must be an object')
    arms = predictions.get('arms')
    if (not isinstance(arms, list) or not arms
            or any(not isinstance(arm, str) or not arm.strip() for arm in arms)
            or len(set(arms)) != len(arms)):
        raise ValueError('arms must be an ordered list of unique nonempty names')
    rows = predictions.get('predictions')
    if not isinstance(rows, list):
        raise ValueError('prediction rows must be a list')
    indexed = {arm: {} for arm in arms}
    for row in rows:
        fields = ('arm', 'stream_id', 'event_id', 'answer')
        if not isinstance(row, dict) or any(not isinstance(row.get(field), str) for field in fields):
            raise ValueError('prediction rows require arm, stream_id, event_id and answer strings')
        arm = row['arm']
        key = row['stream_id'], row['event_id']
        if arm not in indexed or key not in gold:
            raise ValueError(f'unknown prediction arm or query {arm!r} {key!r}')
        if key in indexed[arm]:
            raise ValueError(f'duplicate prediction for {arm!r} {key!r}')
        indexed[arm][key] = row['answer']
    for arm, answers in indexed.items():
        missing = gold.keys() - answers.keys()
        if missing:
            raise ValueError(f'missing predictions for {arm!r}: {sorted(missing)!r}')
    return arms, indexed


def summarize(outcomes):
    count = len(outcomes)
    correct = sum(outcome['correct'] for outcome in outcomes)
    answered = sum(outcome['status'] == 'answered' for outcome in outcomes)
    return {
        'items': count,
        'correct': correct,
        'accuracy': correct / count if count else None,
        'answered': answered,
        'coverage': answered / count if count else None,
        'abstained': sum(outcome['status'] == 'abstained' for outcome in outcomes),
        'blank': sum(outcome['status'] == 'blank' for outcome in outcomes),
    }


def pair_results(items, left, right):
    counts = {'both_correct': 0, 'left_only_correct': 0, 'right_only_correct': 0, 'both_wrong': 0}
    correctness_labels = {
        (True, True): 'both_correct',
        (True, False): 'left_only_correct',
        (False, True): 'right_only_correct',
        (False, False): 'both_wrong',
    }
    disagreements = []
    for item in items:
        first, second = item['outcomes'][left], item['outcomes'][right]
        label = correctness_labels[first['correct'], second['correct']]
        counts[label] += 1
        if first['normalized_answer'] != second['normalized_answer']:
            disagreements.append({'stream_id': item['stream_id'], 'event_id': item['event_id'],
                                  'case': item['case'], 'correctness': label})
    return {'arms': [left, right], **counts, 'answer_disagreements': len(disagreements),
            'disagreements': disagreements}


def score(streams, predictions):
    gold = query_gold(streams)
    arms, indexed = index_predictions(predictions, gold)
    items = []
    for key, reference in gold.items():
        outcomes = {}
        for arm in arms:
            raw = indexed[arm][key]
            answer = normalize_answer(raw)
            status = 'blank' if not answer else 'abstained' if answer == UNKNOWN else 'answered'
            outcomes[arm] = {'raw_answer': raw, 'normalized_answer': answer,
                             'status': status, 'correct': answer == reference['target']}
        items.append({'stream_id': key[0], 'event_id': key[1], **reference, 'outcomes': outcomes})

    cases = dict.fromkeys(item['case'] for item in items)
    by_arm = {}
    for arm in arms:
        by_arm[arm] = {
            'overall': summarize([item['outcomes'][arm] for item in items]),
            'known': summarize([item['outcomes'][arm] for item in items if item['known']]),
            'unknown': summarize([item['outcomes'][arm] for item in items if not item['known']]),
            'by_case': {case: summarize([item['outcomes'][arm] for item in items if item['case'] == case])
                        for case in cases},
        }
    return {'format': 'pbslm.continuous_memory_score.v1', 'query_items': len(items),
            'unknown_label': UNKNOWN, 'arms': arms, 'by_arm': by_arm,
            'paired': [pair_results(items, left, right) for left, right in combinations(arms, 2)],
            'items': items}


def score_files(fixtures, predictions):
    fixture_hash = hashlib.sha256(fixtures.read_bytes()).hexdigest()
    prediction_bytes = predictions.read_bytes()
    data = json.loads(prediction_bytes)
    if not isinstance(data, dict) or data.get('fixture_sha256') != fixture_hash:
        raise ValueError('prediction fixture_sha256 does not match fixture bytes')
    result = score(load_streams(fixtures), data)
    return {**result, 'fixture_sha256': fixture_hash,
            'prediction_sha256': hashlib.sha256(prediction_bytes).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score_files(args.fixtures, args.predictions)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
