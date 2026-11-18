"""Score sealed snapshot runs from the frozen bounded online memory comparison."""

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

from evaluation.bounded_online_compare import FIXTURE_SHA256
from evaluation.continuous_event_protocol import load_streams
from evaluation.score_continuous_memory import UNKNOWN, pair_results, score, score_files


DEFAULT_FIXTURES = Path(__file__).with_name('fixtures') / 'bounded_online_v1.jsonl'
PREDICTION_FORMAT = 'pbslm.bounded_online_predictions.v1'
QUERY_KEY = ('arm', 'stream_id', 'event_id')
COUNT_FIELDS = ('calls', 'input_tokens', 'output_tokens')
TIME_FIELDS = ('model_service_seconds', 'wall_seconds')
MEMORY_COUNT_FIELDS = ('reads', 'writes', 'evictions')
MEMORY_TIME_FIELDS = ('operation_cpu_seconds', 'operation_wall_seconds')
TRUTH_CATEGORIES = (
    'both_correct', 'readout_correct_answer_wrong', 'readout_wrong_answer_correct', 'both_wrong',
)


def read_run(fixtures, predictions, streams):
    answer_score = score_files(fixtures, predictions)
    data = predictions.read_bytes()
    if hashlib.sha256(data).hexdigest() != answer_score['prediction_sha256']:
        raise ValueError(f'prediction file changed while scoring {predictions}')
    run = json.loads(data)
    if run.get('format') != PREDICTION_FORMAT:
        raise ValueError(f'unsupported snapshot prediction format in {predictions}')
    indexed = {tuple(row[field] for field in QUERY_KEY): row for row in run['predictions']}
    for key, row in indexed.items():
        if 'readout' not in row or (row['readout'] is not None and (
            not isinstance(row['readout'], str) or not row['readout'].strip()
        )):
            raise ValueError(f'memory readout must be a nonempty string or null for {key}')

    arms = run['arms']
    readouts = {
        'arms': arms,
        'predictions': [
            {**dict(zip(QUERY_KEY, key)), 'answer': UNKNOWN if row['readout'] is None else row['readout']}
            for key, row in indexed.items()
        ],
    }
    costs = index_costs(run.get('costs'), arms, streams, indexed)
    result = {
        key: answer_score[key] for key in ('fixture_sha256', 'prediction_sha256', 'arms')
    }
    result['config'] = run.get('config')
    result['checkpoint_files_sha256'] = run.get('checkpoint_files_sha256')
    result['raw_predictions'] = raw_prediction_map(indexed, arms)
    result['overall'] = summarize_run(arms, streams, answer_score, score(streams, readouts), indexed, costs)
    result['by_split'] = {}
    for split in dict.fromkeys(stream['split'] for stream in streams):
        subset = [stream for stream in streams if stream['split'] == split]
        stream_ids = {stream['stream_id'] for stream in subset}
        split_answers = subset_predictions(run, stream_ids)
        split_readouts = subset_predictions(readouts, stream_ids)
        result['by_split'][split] = summarize_run(
            arms, subset, score(subset, split_answers), score(subset, split_readouts), indexed, costs,
        )
    return result, indexed


def subset_predictions(run, stream_ids):
    return {
        'arms': run['arms'],
        'predictions': [row for row in run['predictions'] if row['stream_id'] in stream_ids],
    }


def summarize_run(arms, streams, answers, readouts, indexed, costs):
    query_items = answers['items']
    error_counts = {
        arm: dict.fromkeys((*TRUTH_CATEGORIES, 'answer_matches_readout', 'answer_differs_from_readout',
                           'known_missing_readout', 'known_wrong_readout', 'unknown_nonempty_readout'), 0)
        for arm in arms
    }
    truth_category = {
        (True, True): 'both_correct',
        (True, False): 'readout_correct_answer_wrong',
        (False, True): 'readout_wrong_answer_correct',
        (False, False): 'both_wrong',
    }
    for answer_item, readout_item in zip(query_items, readouts['items'], strict=True):
        stream_id, event_id = answer_item['stream_id'], answer_item['event_id']
        for arm in arms:
            answer = answer_item['outcomes'][arm]
            readout = readout_item['outcomes'][arm]
            counts = error_counts[arm]
            counts[truth_category[readout['correct'], answer['correct']]] += 1
            matches = answer['normalized_answer'] == readout['normalized_answer']
            counts['answer_matches_readout' if matches else 'answer_differs_from_readout'] += 1
            raw_readout = indexed[arm, stream_id, event_id]['readout']
            if not readout['correct']:
                if readout_item['known']:
                    counts['known_missing_readout' if raw_readout is None else 'known_wrong_readout'] += 1
                else:
                    counts['unknown_nonempty_readout'] += 1

    stream_ids = [stream['stream_id'] for stream in streams]
    summary = {
        'query_items': len(query_items),
        'streams': len(streams),
        'answers': {field: answers[field] for field in ('by_arm', 'paired')},
        'readouts': {field: readouts[field] for field in ('by_arm', 'paired')},
        'error_taxonomy': error_counts,
        'raw_pairs': [
            compare_rows(
                (item['stream_id'], item['event_id'], indexed[left, item['stream_id'], item['event_id']],
                 indexed[right, item['stream_id'], item['event_id']])
                for item in query_items
            ) | {'arms': [left, right]}
            for left, right in combinations(arms, 2)
        ],
        'resources': {
            arm: summarize_resources([costs[arm, stream_id] for stream_id in stream_ids])
            for arm in arms
        },
    }
    summary['paired_exact_controls'] = paired_exact_controls(arms, answers, readouts, summary)
    return summary


def paired_exact_controls(arms, answers, readouts, summary):
    pbs = 'pbs_slots'
    if pbs not in arms:
        return {}
    result = {}
    for control in ('exact_unbounded', 'exact_bounded'):
        if control not in arms:
            continue
        pbs_accuracy = answers['by_arm'][pbs]['overall']['accuracy']
        exact_accuracy = answers['by_arm'][control]['overall']['accuracy']
        pbs_state_bytes = summary['resources'][pbs]['peak_total_logical_state_bytes']
        exact_state_bytes = summary['resources'][control]['peak_total_logical_state_bytes']
        raw_pair = next(pair for pair in summary['raw_pairs'] if set(pair['arms']) == {pbs, control})
        result[control] = {
            'arms': [pbs, control],
            'pbs_accuracy': pbs_accuracy,
            'exact_accuracy': exact_accuracy,
            'accuracy_difference_percentage_points': (pbs_accuracy - exact_accuracy) * 100,
            'answers': exact_control_counts(pair_results(answers['items'], pbs, control)),
            'readouts': exact_control_counts(pair_results(readouts['items'], pbs, control)),
            'raw_answers_identical': raw_pair['raw_answers_identical'],
            'raw_readouts_identical': raw_pair['raw_readouts_identical'],
            'peak_total_logical_state_bytes': {
                pbs: pbs_state_bytes,
                control: exact_state_bytes,
                'difference': pbs_state_bytes - exact_state_bytes,
                'ratio': pbs_state_bytes / exact_state_bytes if exact_state_bytes else None,
            },
        }
    return result


def exact_control_counts(pair):
    return {
        'both_correct': pair['both_correct'],
        'pbs_only_correct': pair['left_only_correct'],
        'exact_only_correct': pair['right_only_correct'],
        'both_wrong': pair['both_wrong'],
        'normalized_disagreements': pair['answer_disagreements'],
    }


def compare_rows(pairs):
    count = 0
    differences = []
    answer_disagreements = readout_disagreements = 0
    for stream_id, event_id, left, right in pairs:
        count += 1
        changed = {
            field: {'left': left[field], 'right': right[field]}
            for field in ('answer', 'readout') if left[field] != right[field]
        }
        answer_disagreements += 'answer' in changed
        readout_disagreements += 'readout' in changed
        if changed:
            differences.append({'stream_id': stream_id, 'event_id': event_id, **changed})
    return {
        'items': count,
        'raw_answer_disagreements': answer_disagreements,
        'raw_readout_disagreements': readout_disagreements,
        'raw_answers_identical': answer_disagreements == 0,
        'raw_readouts_identical': readout_disagreements == 0,
        'differences': differences,
    }


def raw_prediction_map(indexed, arms):
    result = {arm: {} for arm in arms}
    for (arm, stream_id, event_id), row in indexed.items():
        result[arm].setdefault(stream_id, {})[event_id] = {
            'answer': row['answer'],
            'readout': row['readout'],
        }
    return result


def index_costs(rows, arms, streams, predictions):
    if not isinstance(rows, list):
        raise ValueError('prediction costs must be a list')
    event_ids_by_stream = {
        stream['stream_id']: [event['event_id'] for event in stream['events'] if event['kind'] == 'query']
        for stream in streams
    }
    expected = {(arm, stream_id) for arm in arms for stream_id in event_ids_by_stream}
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('memory'), dict):
            raise ValueError('prediction costs require per-stream memory metrics')
        arm, stream_id = row.get('arm'), row.get('stream_id')
        if not isinstance(arm, str) or not isinstance(stream_id, str):
            raise ValueError('prediction costs require arm and stream IDs')
        key = arm, stream_id
        if key not in expected or key in indexed:
            raise ValueError(f'unknown or duplicate resource metrics for {key}')
        memory = row['memory']
        require_metrics(row, (*COUNT_FIELDS, 'peak_persistent_payload_bytes'), TIME_FIELDS, key)
        require_metrics(
            memory,
            (*MEMORY_COUNT_FIELDS, 'resident_entries', 'persistent_payload_bytes', 'fixed_reader_bytes'),
            MEMORY_TIME_FIELDS, key,
        )
        event_ids = event_ids_by_stream[stream_id]
        if row['calls'] != len(event_ids) or memory['reads'] != row['calls']:
            raise ValueError(f'call and read counts must match fixture query count for {key}')
        if row['peak_persistent_payload_bytes'] < memory['persistent_payload_bytes']:
            raise ValueError(f'peak mutable payload smaller than final payload for {key}')
        query_rows = [predictions[arm, stream_id, event_id] for event_id in event_ids]
        for prediction in query_rows:
            generation = prediction.get('generation')
            if not isinstance(generation, dict):
                raise ValueError(f'generation metrics are required for {(arm, stream_id)}')
            require_metrics(generation, ('input_tokens', 'output_tokens'), (), (arm, stream_id))
            require_metrics(prediction, (), ('service_seconds',), (arm, stream_id))
        if sum(prediction['generation']['input_tokens'] for prediction in query_rows) != row['input_tokens']:
            raise ValueError(f'input token totals do not match query rows for {key}')
        if sum(prediction['generation']['output_tokens'] for prediction in query_rows) != row['output_tokens']:
            raise ValueError(f'output token totals do not match query rows for {key}')
        service_total = math.fsum(prediction['service_seconds'] for prediction in query_rows)
        if not math.isclose(service_total, row['model_service_seconds'], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f'model service time does not match query rows for {key}')
        indexed[key] = row
    if indexed.keys() != expected:
        raise ValueError(f'missing resource metrics for {sorted(expected - indexed.keys())}')
    return indexed


def require_metrics(row, counts, seconds, key):
    for field in counts:
        value = row.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f'nonnegative integer {field} required for {key}')
    for field in seconds:
        value = row.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'finite nonnegative {field} required for {key}')


def summarize_resources(rows):
    memory = [row['memory'] for row in rows]
    return {
        **{field: sum(row[field] for row in rows) for field in COUNT_FIELDS},
        **{field: math.fsum(row[field] for row in rows) for field in TIME_FIELDS},
        **{field: sum(row[field] for row in memory) for field in MEMORY_COUNT_FIELDS},
        **{f'memory_{field}': math.fsum(row[field] for row in memory) for field in MEMORY_TIME_FIELDS},
        'peak_persistent_payload_bytes': max(row['peak_persistent_payload_bytes'] for row in rows),
        'fixed_reader_bytes': max(row['fixed_reader_bytes'] for row in memory),
        'peak_total_logical_state_bytes': max(
            row['peak_persistent_payload_bytes'] + row['memory']['fixed_reader_bytes'] for row in rows
        ),
        'peak_final_resident_entries': max(row['resident_entries'] for row in memory),
    }


def score_snapshots(fixtures, predictions, repeat):
    streams = load_streams(fixtures)
    if hashlib.sha256(fixtures.read_bytes()).hexdigest() != FIXTURE_SHA256:
        raise ValueError('input does not match the frozen benchmark')
    main, first = read_run(fixtures, predictions, streams)
    repeated, second = read_run(fixtures, repeat, streams)
    if main['fixture_sha256'] != FIXTURE_SHA256 or repeated['fixture_sha256'] != FIXTURE_SHA256:
        raise ValueError('fixture changed while scoring')
    if first.keys() != second.keys():
        raise ValueError('rerun must contain the same arm, stream, and event keys')
    if main['checkpoint_files_sha256'] != repeated['checkpoint_files_sha256']:
        raise ValueError('checkpoint hashes differ across the two runs')

    by_arm = {}
    for arm in main['arms']:
        pairs = (
            (key[1], key[2], first[key], second[key])
            for key in sorted(first)
            if key[0] == arm
        )
        by_arm[arm] = compare_rows(pairs)
    return {
        'format': 'pbslm.bounded_online_score.v1',
        'unknown_label': UNKNOWN,
        'fixture_sha256': FIXTURE_SHA256,
        'main': main,
        'repeat': repeated,
        'rerun': {
            'reverse_arm_order': main['arms'] == list(reversed(repeated['arms'])),
            'by_arm': by_arm,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--repeat', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score_snapshots(args.fixtures, args.predictions, args.repeat)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
