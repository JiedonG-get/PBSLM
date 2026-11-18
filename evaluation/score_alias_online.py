"""Score sealed paired predictions from the alias and correction experiment."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from evaluation.continuous_event_protocol import load_streams
from evaluation.score_continuous_memory import UNKNOWN, normalize_answer, score, summarize


PREDICTION_FORMAT = 'pbslm.alias_online_predictions.v1'
ARMS = ('history', 'lexical', 'structured', 'structured_fallback')
STRUCTURED_ARMS = ('structured', 'structured_fallback')
REPRODUCIBLE_FIELDS = ('answer', 'entity', 'readout', 'evidence_event_ids', 'messages', 'fallback')
CORE_COSTS = ('calls', 'input_tokens', 'output_tokens', 'model_seconds')
COST_FIELDS = (*CORE_COSTS, 'total_seconds', 'host_seconds', 'cpu_seconds',
               'maintenance_calls', 'maintenance_input_tokens', 'maintenance_output_tokens',
               'maintenance_model_seconds', 'linker_calls', 'answer_calls', 'fallback_queries',
               'peak_state_bytes', 'answer_input_tokens', 'answer_output_tokens', 'answer_model_seconds')
TRACE_FIELDS = ('validated_operations', 'applied_operations', 'rejected_parse', 'rejected_state')


def _query_references(streams):
    references = {}
    for stream in streams:
        prior = set()
        for event in stream['events']:
            if event['kind'] == 'observation':
                prior.add(event['event_id'])
                continue
            key = stream['stream_id'], event['event_id']
            oracle = event.get('oracle')
            if not isinstance(oracle, dict) or 'entity' not in oracle:
                raise ValueError(f'missing alias query oracle for {key}')
            entity = oracle['entity']
            if entity is not None and (not isinstance(entity, str) or not normalize_answer(entity)):
                raise ValueError(f'invalid query oracle entity for {key}')
            evidence = _evidence_ids(oracle.get('evidence_event_ids'), prior, key)
            references[key] = {'prior': frozenset(prior), 'evidence': evidence, 'entity': entity}
    return references


def _evidence_ids(value, prior, key):
    if (not isinstance(value, list) or any(not isinstance(event_id, str) for event_id in value)
            or len(set(value)) != len(value) or not prior.issuperset(value)):
        raise ValueError(f'evidence must be unique IDs of prior observations for {key}')
    return frozenset(value)


def _index_rows(data, references):
    indexed = {arm: {} for arm in data['arms']}
    for row in data['predictions']:
        arm = row['arm']
        key = row['stream_id'], row['event_id']
        _evidence_ids(row.get('evidence_event_ids'), references[key]['prior'], (arm, *key))
        for field in ('entity', 'readout'):
            if field not in row or (row[field] is not None and (
                    arm not in STRUCTURED_ARMS or not isinstance(row[field], str)
                    or not normalize_answer(row[field]))):
                raise ValueError(f'invalid {field} for {(arm, *key)}')
        if not isinstance(row.get('messages'), list):
            raise ValueError(f'messages must be a list for {(arm, *key)}')
        if arm == 'structured_fallback' and type(row.get('fallback')) is not bool:
            raise ValueError(f'fallback must be boolean for {(arm, *key)}')
        indexed[arm][key] = row
    return indexed


def _index_costs(rows, arms, streams):
    if not isinstance(rows, list):
        raise ValueError('costs must be a list')
    expected = {(arm, stream['stream_id']) for arm in arms for stream in streams}
    costs = {}
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(field), str) for field in ('arm', 'stream_id')):
            raise ValueError('costs require arm and stream_id strings')
        key = row['arm'], row['stream_id']
        if key not in expected or key in costs:
            raise ValueError(f'unexpected or duplicate costs for {key}')
        for field in COST_FIELDS:
            if field not in row and field not in CORE_COSTS:
                continue
            value = row.get(field)
            if field.endswith('_seconds'):
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            else:
                valid = type(value) is int and value >= 0
            if not valid:
                raise ValueError(f'invalid {field} for {key}')
        costs[key] = row
    if costs.keys() != expected:
        raise ValueError(f'missing costs for {sorted(expected - costs.keys())}')
    return costs


def _resources(rows):
    totals = {}
    for field in COST_FIELDS:
        values = [row[field] for row in rows if field in row]
        if len(values) != len(rows):
            continue
        if field == 'peak_state_bytes':
            totals[field] = max(values)
        elif field.endswith('_seconds'):
            totals[field] = math.fsum(values)
        else:
            totals[field] = sum(values)
    return totals


def _summarize_traces(traces, arms, costs):
    if not isinstance(traces, list):
        raise ValueError('ingress traces must be a list')
    ingresses = {arm: Counter(dict.fromkeys(TRACE_FIELDS, 0)) for arm in arms}
    call_hashes = {}
    for trace in traces:
        key = trace['arm'], trace['stream_id']
        if key not in costs or key in call_hashes or len(trace['calls']) != costs[key]['calls']:
            raise ValueError(f'unknown, duplicate, or inconsistent call trace for {key}')
        call_rows = [{field: call[field] for field in ('event_id', 'purpose', 'messages', 'raw_output')}
                     for call in trace['calls']]
        call_hashes[key] = hashlib.sha256(json.dumps(call_rows, ensure_ascii=False, sort_keys=True,
                                                      separators=(',', ':')).encode('utf-8')).hexdigest()
        counts = ingresses[key[0]]
        counts['records'] += len(trace['ingress'])
        for event in trace['ingress']:
            for field in TRACE_FIELDS:
                counts[field] += len(event[field])
    if call_hashes.keys() != costs.keys():
        raise ValueError(f'missing call trace for {sorted(costs.keys() - call_hashes.keys())}')
    return {arm: dict(counts) for arm, counts in ingresses.items()}, call_hashes


def _outcomes(arm, items, references, rows):
    diagnostics = []
    for item in items:
        key = item['stream_id'], item['event_id']
        ref, row = references[key], rows[key]
        actual = item['outcomes'][arm]
        evidence = set(row['evidence_event_ids'])
        diagnostic = {'stream_id': key[0], 'event_id': key[1], 'case': item['case'],
                      'known': item['known'], 'answer': actual,
                      'over_abstained': item['known'] and actual['status'] == 'abstained',
                      'under_abstained': not item['known'] and actual['status'] == 'answered',
                      'available_prior_observations': len(ref['prior']),
                      'evidence_event_ids': row['evidence_event_ids'],
                      'gold_evidence_event_ids': sorted(ref['evidence']),
                      'gold_links_found': len(evidence & ref['evidence']),
                      'gold_evidence_complete': ref['evidence'] <= evidence if ref['evidence'] else None}
        if arm in STRUCTURED_ARMS:
            entity = normalize_answer(row['entity']) if row['entity'] is not None else None
            gold_entity = normalize_answer(ref['entity']) if ref['entity'] is not None else None
            readout = normalize_answer(row['readout']) if row['readout'] is not None else UNKNOWN
            error = 'correct'
            if readout != item['target']:
                if not item['known']:
                    error = 'unknown_nonempty_readout'
                else:
                    error = 'known_missing_readout' if readout == UNKNOWN else 'known_wrong_readout'
            diagnostic.update({'entity': row['entity'], 'gold_entity': ref['entity'],
                               'entity_correct': entity == gold_entity, 'readout': row['readout'],
                               'readout_error': error, 'readout_correct': error == 'correct',
                               'answer_matches_readout': actual['normalized_answer'] == readout})
        if arm == 'structured_fallback':
            diagnostic['fallback'] = row['fallback']
        diagnostics.append(diagnostic)
    return diagnostics


def _diagnostic_summary(arm, rows):
    evidence_rows = [row for row in rows if row['gold_evidence_complete'] is not None]
    complete = [row for row in evidence_rows if row['gold_evidence_complete']]
    incomplete = [row for row in evidence_rows if not row['gold_evidence_complete']]
    totals = {'over_abstained': sum(row['over_abstained'] for row in rows),
              'under_abstained': sum(row['under_abstained'] for row in rows),
              'evidence': {'queries_with_gold_links': len(evidence_rows), 'complete': len(complete),
                           'gold_links': sum(len(row['gold_evidence_event_ids']) for row in evidence_rows),
                           'gold_links_found': sum(row['gold_links_found'] for row in evidence_rows),
                           'selected_rows': sum(len(row['evidence_event_ids']) for row in rows),
                           'answers_with_complete_evidence': summarize([row['answer'] for row in complete]),
                           'answers_with_incomplete_evidence': summarize([row['answer'] for row in incomplete])}}
    if arm in ('history', 'lexical'):
        totals['evidence']['prefixes_with_omissions'] = sum(
            len(row['evidence_event_ids']) < row['available_prior_observations'] for row in rows)
    if arm in STRUCTURED_ARMS:
        gold_entities = [row for row in rows if row['gold_entity'] is not None]
        totals['entities'] = {'items': len(rows), 'correct': sum(row['entity_correct'] for row in rows),
                              'known': len(gold_entities),
                              'known_correct': sum(row['entity_correct'] for row in gold_entities),
                              'known_unresolved': sum(row['entity'] is None for row in gold_entities),
                              'unknown_overresolved': sum(row['entity'] is not None for row in rows
                                                          if row['gold_entity'] is None)}
        totals['readout'] = {'items': len(rows), 'correct': sum(row['readout_correct'] for row in rows),
                             'errors': dict(Counter(row['readout_error'] for row in rows if not row['readout_correct'])),
                             'answer_matches_readout': sum(row['answer_matches_readout'] for row in rows),
                             'readout_correct_answer_wrong': sum(row['readout_correct'] and not row['answer']['correct']
                                                                 for row in rows),
                             'readout_wrong_answer_correct': sum(not row['readout_correct'] and row['answer']['correct']
                                                                 for row in rows)}
    if arm == 'structured_fallback':
        totals['fallback_answers'] = summarize([row['answer'] for row in rows if row['fallback']])
    return totals


def _summarize(streams, answer_score, rows, references, costs):
    arms = answer_score['arms']
    items = answer_score['items']
    diagnostics = {}
    resources = {}
    for arm in arms:
        outcomes = _outcomes(arm, items, references, rows[arm])
        cases = dict.fromkeys(outcome['case'] for outcome in outcomes)
        diagnostics[arm] = {'overall': _diagnostic_summary(arm, outcomes),
                            'by_case': {case: _diagnostic_summary(arm, [row for row in outcomes if row['case'] == case])
                                        for case in cases},
                            'items': outcomes}
        resources[arm] = _resources([costs[arm, stream['stream_id']] for stream in streams])
    return {'streams': len(streams), 'query_items': len(items),
            'answers': {field: answer_score[field] for field in ('by_arm', 'paired')},
            'diagnostics': diagnostics, 'resources': resources}


def _read_run(path, streams, fixture_hash, references):
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get('format') != PREDICTION_FORMAT:
        raise ValueError(f'unsupported alias prediction format in {path}')
    if data.get('fixture_sha256') != fixture_hash:
        raise ValueError(f'prediction fixture_sha256 does not match fixture bytes in {path}')
    answer_score = score(streams, data)
    arms = answer_score['arms']
    if set(arms) != set(ARMS):
        raise ValueError(f'expected all four alias experiment arms in {path}')
    rows = _index_rows(data, references)
    costs = _index_costs(data.get('costs'), arms, streams)
    run = {'prediction_sha256': hashlib.sha256(raw).hexdigest(), 'arms': arms,
           'config': data.get('config'), 'checkpoint_files_sha256': data.get('checkpoint_files_sha256'),
           'overall': _summarize(streams, answer_score, rows, references, costs)}
    by_split = {}
    for split in dict.fromkeys(stream.get('split', 'unspecified') for stream in streams):
        subset = [stream for stream in streams if stream.get('split', 'unspecified') == split]
        stream_ids = {stream['stream_id'] for stream in subset}
        predictions = [row for row in data['predictions'] if row['stream_id'] in stream_ids]
        by_split[split] = _summarize(subset, score(subset, {'arms': arms, 'predictions': predictions}),
                                     rows, references, costs)
    run['by_split'] = by_split
    call_hashes = None
    if 'ingress' in data:
        run['ingress'], call_hashes = _summarize_traces(data['ingress'], arms, costs)
    return run, rows, call_hashes


def _compare_repeats(main, repeated, first, second, first_call_hashes, second_call_hashes):
    if main['checkpoint_files_sha256'] != repeated['checkpoint_files_sha256']:
        raise ValueError('checkpoint hashes differ across the two runs')
    by_arm = {}
    for arm in main['arms']:
        counts = dict.fromkeys(REPRODUCIBLE_FIELDS, 0)
        differences = []
        for (stream_id, event_id), left in first[arm].items():
            right = second[arm][stream_id, event_id]
            changed = [field for field in REPRODUCIBLE_FIELDS if left.get(field) != right.get(field)]
            for field in changed:
                counts[field] += 1
            if changed:
                differences.append({'stream_id': stream_id, 'event_id': event_id, 'fields': changed})
        by_arm[arm] = {'items': len(first[arm]), 'identical': not differences,
                       'disagreements_by_field': counts, 'differences': differences}
    result = {'reverse_arm_order': main['arms'] == list(reversed(repeated['arms'])), 'by_arm': by_arm}
    if first_call_hashes is not None and second_call_hashes is not None:
        result['streams_with_call_disagreements'] = {
            arm: [stream_id for key_arm, stream_id in first_call_hashes
                  if key_arm == arm and first_call_hashes[arm, stream_id] != second_call_hashes[arm, stream_id]]
            for arm in main['arms']}
    return result


def score_alias_snapshots(fixtures, predictions, repeat=None):
    fixture_hash = hashlib.sha256(fixtures.read_bytes()).hexdigest()
    streams = load_streams(fixtures)
    if hashlib.sha256(fixtures.read_bytes()).hexdigest() != fixture_hash:
        raise ValueError('fixture changed while loading')
    references = _query_references(streams)
    main, first, first_call_hashes = _read_run(predictions, streams, fixture_hash, references)
    result = {'format': 'pbslm.alias_online_score.v1', 'unknown_label': UNKNOWN,
              'fixture_sha256': fixture_hash, 'main': main}
    if repeat is not None:
        repeated, second, second_call_hashes = _read_run(repeat, streams, fixture_hash, references)
        result.update({'repeat': repeated, 'rerun': _compare_repeats(
            main, repeated, first, second, first_call_hashes, second_call_hashes,
        )})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--repeat', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score_alias_snapshots(args.fixtures, args.predictions, args.repeat)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
