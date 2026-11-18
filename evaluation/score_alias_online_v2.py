"""Score paired v2 alias predictions, state statuses, traces, costs and repeats."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path

from evaluation.continuous_event_protocol import load_streams
from evaluation.alias_online_compare_v2 import EXTRACT_PROMPT as SCOPED_EXTRACT_PROMPT
from evaluation.score_alias_online import TRACE_FIELDS
from evaluation.score_continuous_memory import UNKNOWN, normalize_answer, pair_results, score as score_answers, summarize


PREDICTION_FORMAT = 'pbslm.alias_online_predictions.v2'
SCORE_FORMAT = 'pbslm.alias_online_score.v2'
ARMS = ('history', 'lexical', 'structured', 'scoped')
READOUT_ARMS = ('structured', 'scoped')
ENTITY_STATUSES = {'resolved', 'unknown', 'ambiguous'}
LOCATION_STATUSES = {'located', 'no_location', 'missing'}
REPRODUCIBLE_FIELDS = (
    'answer', 'entity', 'entity_status', 'location_status', 'readout', 'evidence_event_ids',
    'memory_evidence_event_ids', 'messages', 'fallback', 'link_method', 'available_observations',
)
CALL_FIELDS = ('event_id', 'purpose', 'messages', 'raw_output', 'generation')
SCOPED_BATCH_FIELDS = ('query_event_id', 'observation_event_ids')
PURPOSES = ('maintenance', 'link', 'answer')
CORE_COSTS = ('calls', 'input_tokens', 'output_tokens', 'model_seconds')
PURPOSE_COST_FIELDS = {
    purpose: (count, f'{purpose}_input_tokens', f'{purpose}_output_tokens', f'{purpose}_model_seconds')
    for purpose, count in zip(PURPOSES, ('maintenance_calls', 'linker_calls', 'answer_calls'))
}
COST_FIELDS = (
    *CORE_COSTS, 'total_seconds', 'host_seconds', 'fallback_queries', 'peak_state_bytes',
    *(field for fields in PURPOSE_COST_FIELDS.values() for field in fields),
)


def _evidence_ids(value, prior, where):
    if (not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value)) or not prior.issuperset(value)):
        raise ValueError(f'evidence IDs must be unique prior observations for {where}')
    return frozenset(value)


def query_references(streams):
    references = {}
    for stream in streams:
        prior = set()
        for event in stream['events']:
            if event['kind'] == 'observation':
                prior.add(event['event_id'])
                continue
            key = (stream['stream_id'], event['event_id'])
            oracle = event.get('oracle')
            if not isinstance(oracle, dict) or 'entity' not in oracle:
                raise ValueError(f'missing alias query oracle for {key}')
            raw_target = oracle.get('target')
            target = normalize_answer(raw_target) if isinstance(raw_target, str) else ''
            entity = oracle.get('entity')
            entity_state = oracle.get('entity_state')
            state = oracle.get('state')
            case = oracle.get('case')
            if (not target or not isinstance(state, str) or state not in LOCATION_STATUSES
                    or not isinstance(entity_state, str) or entity_state not in ENTITY_STATUSES
                    or not isinstance(case, str) or not case.strip()
                    or (entity is not None and (not isinstance(entity, str) or not normalize_answer(entity)))):
                raise ValueError(f'invalid alias state oracle for {key}')
            if (entity_state == 'resolved') != (entity is not None):
                raise ValueError(f'entity and entity_state disagree for {key}')
            if entity is None and state != 'missing':
                raise ValueError(f'unresolved query must have missing location for {key}')
            if (state == 'located') != (target != UNKNOWN):
                raise ValueError(f'query target conflicts with location state for {key}')
            evidence = _evidence_ids(oracle.get('evidence_event_ids'), prior, key)
            references[key] = {
                'prior': frozenset(prior), 'evidence': evidence,
                'entity': entity, 'entity_state': entity_state, 'state': state,
                'target': target, 'case': case,
            }
    if not references:
        raise ValueError('fixture must contain query events')
    return references


def _index_predictions(data, references):
    rows = data.get('predictions')
    if not isinstance(rows, list):
        raise ValueError('predictions must be a list')
    indexed = {arm: {} for arm in data['arms']}
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(field), str) for field in ('arm', 'stream_id', 'event_id')):
            raise ValueError('prediction rows require arm, stream_id and event_id strings')
        arm = row.get('arm')
        key = row.get('stream_id'), row.get('event_id')
        if arm not in indexed or key not in references or key in indexed[arm]:
            raise ValueError(f'unknown or duplicate prediction for {(arm, *key)}')
        if not isinstance(row.get('answer'), str):
            raise ValueError(f'answer must be a string for {(arm, *key)}')
        if not isinstance(row.get('messages'), list):
            raise ValueError(f'messages must be a list for {(arm, *key)}')
        _evidence_ids(row.get('evidence_event_ids'), references[key]['prior'], (arm, *key))
        memory_evidence = row.get('memory_evidence_event_ids', [])
        _evidence_ids(memory_evidence, references[key]['prior'], (arm, *key, 'memory'))
        if arm in READOUT_ARMS:
            if 'entity' not in row or 'readout' not in row:
                raise ValueError(f'missing entity or readout field for {(arm, *key)}')
            entity, readout = row['entity'], row['readout']
            if entity is not None and (not isinstance(entity, str) or not normalize_answer(entity)):
                raise ValueError(f'invalid entity for {(arm, *key)}')
            if readout is not None and (not isinstance(readout, str) or not normalize_answer(readout)):
                raise ValueError(f'invalid readout for {(arm, *key)}')
            if arm == 'scoped':
                entity_status, location_status = row.get('entity_status'), row.get('location_status')
                if 'location_status' not in row:
                    raise ValueError(f'missing location_status for {(arm, *key)}')
                if not isinstance(entity_status, str) or entity_status not in ENTITY_STATUSES:
                    raise ValueError(f'invalid entity_status for {(arm, *key)}')
                if (entity_status == 'resolved') != (entity is not None):
                    raise ValueError(f'entity and entity_status disagree for {(arm, *key)}')
                if location_status is not None and (
                        not isinstance(location_status, str) or location_status not in LOCATION_STATUSES):
                    raise ValueError(f'invalid location_status for {(arm, *key)}')
                if entity_status != 'resolved' and (
                        entity is not None or location_status is not None or readout is not None):
                    raise ValueError(f'unresolved entity requires null location status and readout for {(arm, *key)}')
                if entity_status == 'resolved' and location_status is None:
                    raise ValueError(f'resolved entity requires a location status for {(arm, *key)}')
                if (location_status == 'located') != (readout is not None):
                    raise ValueError(f'located status and readout disagree for {(arm, *key)}')
                expected_answer = readout if readout is not None else UNKNOWN
                if row['messages'] or row['answer'] != expected_answer:
                    raise ValueError(f'scoped answer must match the direct state readout for {(arm, *key)}')
            elif any(field in row for field in ('entity_status', 'location_status')):
                raise ValueError(f'v1 structured arm must not supply v2 statuses for {(arm, *key)}')
        query_seconds = row.get('query_seconds')
        if (type(query_seconds) not in (int, float) or not math.isfinite(query_seconds)
                or query_seconds < 0):
            raise ValueError(f'query_seconds must be finite and nonnegative for {(arm, *key)}')
        if arm == 'scoped' and row.get('fallback') is not False:
            raise ValueError(f'scoped arm must not silently use a fallback for {(arm, *key)}')
        if arm in {'history', 'lexical'}:
            if any(field in row for field in ('entity_status', 'location_status')):
                raise ValueError(f'baseline arm must not supply state-classifier outputs for {(arm, *key)}')
        indexed[arm][key] = row

    expected = set(references)
    for arm, arm_rows in indexed.items():
        if arm_rows.keys() != expected:
            raise ValueError(f'prediction coverage mismatch for {arm}: missing={sorted(expected - arm_rows.keys())}, '
                             f'extra={sorted(arm_rows.keys() - expected)}')
    return indexed


def _index_costs(data, arms, streams):
    rows = data.get('costs')
    if not isinstance(rows, list):
        raise ValueError('costs must be a list')
    expected = {(arm, stream['stream_id']) for arm in arms for stream in streams}
    costs = {}
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(field), str) for field in ('arm', 'stream_id')):
            raise ValueError('cost rows require arm and stream_id')
        key = row['arm'], row['stream_id']
        if key not in expected or key in costs:
            raise ValueError(f'unexpected or duplicate cost row for {key}')
        for field in COST_FIELDS:
            if field not in row:
                if field in CORE_COSTS:
                    raise ValueError(f'missing required cost {field} for {key}')
                continue
            value = row[field]
            if field.endswith('_seconds'):
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            else:
                valid = type(value) is int and value >= 0
            if not valid:
                raise ValueError(f'invalid cost {field} for {key}')
        costs[key] = row
    if costs.keys() != expected:
        raise ValueError(f'missing cost rows for {sorted(expected - costs.keys())}')
    return costs


def _resources(rows):
    result = {}
    keys = set.intersection(*(set(row) for row in rows)) if rows else set()
    for field in sorted(keys & set(COST_FIELDS)):
        values = [row[field] for row in rows]
        if field.startswith('peak_'):
            result[field] = max(values)
        elif field.endswith('_seconds'):
            result[field] = math.fsum(values)
        else:
            result[field] = sum(values)
    return result


def _readout_diagnostic(row, reference, answer, *, scoped):
    entity = normalize_answer(row['entity']) if row['entity'] is not None else None
    gold_entity = normalize_answer(reference['entity']) if reference['entity'] is not None else None
    entity_correct = entity == gold_entity
    actual_readout = normalize_answer(row['readout']) if row['readout'] is not None else None
    expected_readout = reference['target'] if reference['state'] == 'located' else None
    readout_correct = actual_readout == expected_readout
    result = {
        'entity': row['entity'], 'gold_entity': reference['entity'],
        'entity_correct': entity_correct, 'readout': row['readout'],
        'expected_readout': expected_readout, 'readout_correct': readout_correct,
        'entity_and_readout_correct': entity_correct and readout_correct,
        'readout_error': ('correct' if readout_correct else
                          'known_missing_readout' if expected_readout is not None and actual_readout is None else
                          'known_wrong_readout' if expected_readout is not None else 'unexpected_nonempty_readout'),
        'answer_matches_readout': answer['normalized_answer'] == (actual_readout or UNKNOWN),
    }
    if scoped:
        gold_entity_resolved = reference['entity_state'] == 'resolved'
        expected_location_status = reference['state'] if gold_entity_resolved else None
        result.update({
            'entity_status': row['entity_status'], 'gold_entity_status': reference['entity_state'],
            'entity_status_correct': row['entity_status'] == reference['entity_state'],
            'entity_status_compatible': (row['entity_status'] == 'resolved') if gold_entity_resolved
                                        else row['entity'] is None and row['entity_status'] in {'unknown', 'ambiguous'},
            'gold_entity_resolved': gold_entity_resolved,
            'location_status': row['location_status'], 'gold_location_status': expected_location_status,
            'location_status_correct': row['location_status'] == expected_location_status,
            'entity_location_readout_correct': (
                entity_correct and readout_correct and row['entity_status'] == reference['entity_state']
                and row['location_status'] == expected_location_status
            ),
        })
    return result


def _score_items(streams, data, references, indexed):
    answer_score = score_answers(streams, data)
    items = []
    for item in answer_score['items']:
        key = item['stream_id'], item['event_id']
        ref = references[key]
        evidence_gold = ref['evidence']
        outcomes = {}
        for arm in indexed:
            row = indexed[arm][key]
            outcome = item['outcomes'][arm]
            selected = frozenset(row['evidence_event_ids'])
            diagnostic = {
                **outcome, 'evidence_event_ids': row['evidence_event_ids'],
                'gold_evidence_event_ids': sorted(evidence_gold),
                'gold_evidence_complete': evidence_gold <= selected if evidence_gold else None,
                'gold_links_found': len(evidence_gold & selected),
                'available_prior_observations': len(ref['prior']),
            }
            if arm in READOUT_ARMS:
                diagnostic.update(_readout_diagnostic(row, ref, outcome, scoped=arm == 'scoped'))
            if arm == 'scoped':
                diagnostic['fallback'] = row.get('fallback', False)
                diagnostic['link_method'] = row.get('link_method')
            outcomes[arm] = diagnostic
        items.append({**item, 'gold_state': ref['state'], 'gold_entity': ref['entity'],
                      'gold_entity_state': ref['entity_state'], 'outcomes': outcomes})
    return items


def _summarize_diagnostics(items, arm):
    rows = [item['outcomes'][arm] for item in items]
    evidence_rows = [row for row in rows if row['gold_evidence_event_ids']]
    complete_evidence = [row for row in evidence_rows if row['gold_evidence_complete']]
    incomplete_evidence = [row for row in evidence_rows if not row['gold_evidence_complete']]
    result = {
        'over_abstained': sum(item['known'] and item['outcomes'][arm]['status'] == 'abstained' for item in items),
        'under_abstained': sum(not item['known'] and item['outcomes'][arm]['status'] == 'answered' for item in items),
        'evidence': {
            'items_with_gold_links': sum(bool(row['gold_evidence_event_ids']) for row in rows),
            'complete': sum(row['gold_evidence_complete'] for row in rows if row['gold_evidence_event_ids']),
            'gold_links': sum(len(row['gold_evidence_event_ids']) for row in rows),
            'gold_links_found': sum(row['gold_links_found'] for row in rows),
            'selected_rows': sum(len(row['evidence_event_ids']) for row in rows),
            'answers_with_complete_evidence': summarize(complete_evidence),
            'answers_with_incomplete_evidence': summarize(incomplete_evidence),
        },
    }
    if arm in {'history', 'lexical'}:
        result['evidence']['prefixes_with_omissions'] = sum(
            len(row['evidence_event_ids']) < row['available_prior_observations'] for row in rows
        )
    if arm in READOUT_ARMS:
        result.update({
            'entities': {
                'items': len(rows), 'correct': sum(row['entity_correct'] for row in rows),
                'known': sum(row['gold_entity'] is not None for row in rows),
                'known_correct': sum(row['gold_entity'] is not None and row['entity_correct'] for row in rows),
                'known_unresolved': sum(row['gold_entity'] is not None and row['entity'] is None for row in rows),
                'unknown_overresolved': sum(row['gold_entity'] is None and row['entity'] is not None for row in rows),
            },
            'readout': {
                'items': len(rows), 'correct': sum(row['readout_correct'] for row in rows),
                'entity_and_readout_correct': sum(row['entity_and_readout_correct'] for row in rows),
                'errors': dict(Counter(row['readout_error'] for row in rows if row['readout_error'] != 'correct')),
                'answer_matches_readout': sum(row['answer_matches_readout'] for row in rows),
                'readout_correct_answer_wrong': sum(row['readout_correct'] and not row['correct'] for row in rows),
                'readout_wrong_answer_correct': sum(not row['readout_correct'] and row['correct'] for row in rows),
            },
        })
    if arm == 'scoped':
        result['entities'].update({
            'status_correct': sum(row['entity_status_correct'] for row in rows),
            'status_compatible': sum(row['entity_status_compatible'] for row in rows),
            'gold_resolved': sum(row['gold_entity_resolved'] for row in rows),
            'unresolved_gold': sum(not row['gold_entity_resolved'] for row in rows),
            'unknown_predictions_on_unresolved': sum(
                not row['gold_entity_resolved'] and row['entity_status'] == 'unknown' for row in rows),
            'ambiguous_predictions_on_unresolved': sum(
                not row['gold_entity_resolved'] and row['entity_status'] == 'ambiguous' for row in rows),
            'by_gold_status': {
                state: {
                    'items': sum(row['gold_entity_status'] == state for row in rows),
                    'correct': sum(row['gold_entity_status'] == state and row['entity_status_correct']
                                   for row in rows),
                }
                for state in sorted(ENTITY_STATUSES)
            },
        })
        result['location_status'] = {
            'items': len(rows), 'correct': sum(row['location_status_correct'] for row in rows),
            'by_gold': {state or 'unresolved': {'items': sum(row['gold_location_status'] == state for row in rows),
                                               'correct': sum(row['gold_location_status'] == state
                                                              and row['location_status_correct'] for row in rows)}
                        for state in (*sorted(LOCATION_STATUSES), None)},
        }
        result['fallback_items'] = sum(row['fallback'] for row in rows)
        result['entity_location_readout_correct'] = sum(row['entity_location_readout_correct'] for row in rows)
    return result


def _aggregate(streams, items, costs, arms):
    cases = dict.fromkeys(item['case'] for item in items)
    by_arm, resources = {}, {}
    for arm in arms:
        by_arm[arm] = {
            'overall': summarize([item['outcomes'][arm] for item in items]),
            'known': summarize([item['outcomes'][arm] for item in items if item['known']]),
            'unknown': summarize([item['outcomes'][arm] for item in items if not item['known']]),
            'by_case': {case: summarize([item['outcomes'][arm] for item in items if item['case'] == case])
                        for case in cases},
        }
        resources[arm] = _resources([costs[arm, stream['stream_id']] for stream in streams])
    diagnostic = {
        arm: {'overall': _summarize_diagnostics(items, arm),
              'by_case': {case: _summarize_diagnostics([item for item in items if item['case'] == case], arm)
                          for case in cases}}
        for arm in arms
    }
    return by_arm, diagnostic, resources


def _query_latency(indexed_predictions, arms, stream_ids=None):
    allowed_streams = None if stream_ids is None else set(stream_ids)
    summaries = {}
    for arm in arms:
        values = sorted(
            row['query_seconds']
            for (stream_id, _), row in indexed_predictions[arm].items()
            if allowed_streams is None or stream_id in allowed_streams
        )
        count = len(values)
        median = None
        if count:
            middle = count // 2
            median = values[middle] if count % 2 else math.fsum(values[middle - 1:middle + 1]) / 2
        summaries[arm] = {
            'queries': count,
            'total_seconds': math.fsum(values),
            'mean_seconds': math.fsum(values) / count if count else None,
            'median_seconds': median,
            'p90_seconds': values[math.ceil(0.9 * count) - 1] if count else None,
            'min_seconds': values[0] if values else None,
            'max_seconds': values[-1] if values else None,
        }
    return summaries


def _state_pair(items):
    metrics = {
        'entity_correct': ('entity',),
        'readout_correct': ('readout',),
        'entity_and_readout_correct': ('entity', 'readout'),
    }
    paired = {}
    for metric, fields in metrics.items():
        projected = [
            {**item, 'outcomes': {
                arm: {
                    'correct': item['outcomes'][arm][metric],
                    'normalized_answer': tuple(
                        normalize_answer(item['outcomes'][arm][field]) if item['outcomes'][arm][field] is not None
                        else UNKNOWN if field == 'readout' else None
                        for field in fields
                    ),
                }
                for arm in READOUT_ARMS
            }}
            for item in items
        ]
        result = pair_results(projected, *READOUT_ARMS)
        result['output_disagreements'] = result.pop('answer_disagreements')
        paired[metric] = result
    return paired


def _validate_scoped_batches(stream, ingress, calls, key):
    selected_ids = set()
    for row in ingress:
        skipped = row.get('maintenance_skipped')
        if type(skipped) is not bool:
            raise ValueError(f'scoped ingress requires a boolean maintenance_skipped for {key}')
        if skipped and any(row[field] for field in TRACE_FIELDS):
            raise ValueError(f'skipped scoped ingress must have empty operation lists for {key}')
        if not skipped:
            selected_ids.add(row['event_id'])

    pending = []
    expected = []
    for event in stream['events']:
        event_id = event['event_id']
        if event['kind'] == 'observation':
            if event_id in selected_ids:
                pending.append(event_id)
        else:
            if pending:
                expected.append((event_id, pending))
            pending = []
    if pending:
        raise ValueError(f'trailing observations must be skipped for {key}')

    actual = []
    for call in calls:
        query_id = call.get('query_event_id')
        if call['purpose'] != 'maintenance' or call['event_id'] != query_id:
            raise ValueError(f'scoped model calls must be maintenance batches keyed by query for {key}')
        actual.append((query_id, call.get('observation_event_ids')))
    if actual != expected:
        raise ValueError(f'scoped batch observations must match non-skipped ingress in pending order for {key}')

    observations = {event['event_id']: event for event in stream['events']
                    if event['kind'] == 'observation'}
    calls_by_query = {call['query_event_id']: call for call in calls}
    for query_id, observation_ids in expected:
        batch = [{'event_id': event_id, 'text': observations[event_id]['text']}
                 for event_id in observation_ids]
        expected_messages = [
            {'role': 'system', 'content': SCOPED_EXTRACT_PROMPT},
            {'role': 'user', 'content': json.dumps(batch, ensure_ascii=False)},
        ]
        if calls_by_query[query_id]['messages'] != expected_messages:
            raise ValueError(f'scoped maintenance prompt differs from its declared fixture batch for {key}')


def _validate_traces(data, streams, costs, arms, indexed_predictions):
    traces = data.get('ingress')
    if not isinstance(traces, list):
        raise ValueError('v2 predictions require ingress and model-call traces')
    stream_map = {stream['stream_id']: stream for stream in streams}
    expected = {(arm, stream['stream_id']) for arm in arms for stream in streams}
    indexed = {}
    by_arm = {arm: {'observations': 0, 'applied_operations': 0, 'rejected_operations': 0,
                    'calls': 0, 'maintenance_calls': 0, 'link_calls': 0, 'answer_calls': 0} for arm in arms}
    hashes = {}
    for trace in traces:
        if not isinstance(trace, dict):
            raise ValueError('ingress trace rows must be objects')
        if not isinstance(trace.get('arm'), str) or not isinstance(trace.get('stream_id'), str):
            raise ValueError('ingress traces require arm and stream_id strings')
        key = trace.get('arm'), trace.get('stream_id')
        if key not in expected or key in indexed:
            raise ValueError(f'unexpected or duplicate ingress trace {key}')
        stream = stream_map[key[1]]
        observation_ids = [event['event_id'] for event in stream['events'] if event['kind'] == 'observation']
        ingress = trace.get('ingress')
        calls = trace.get('calls')
        if not isinstance(ingress, list) or not isinstance(calls, list):
            raise ValueError(f'ingress and calls must be lists for {key}')
        expected_ingress = observation_ids if key[0] in {'structured', 'scoped'} else []
        if any(not isinstance(row, dict) for row in ingress):
            raise ValueError(f'ingress rows must be objects for {key}')
        if [row.get('event_id') for row in ingress] != expected_ingress:
            raise ValueError(f'ingress event sequence differs from fixture for {key}')
        for row in ingress:
            if any(not isinstance(row.get(field), list) for field in TRACE_FIELDS):
                raise ValueError(f'operation traces must be lists for {key}')
            applied = row['applied_operations']
            rejected_parse = row['rejected_parse']
            rejected_state = row['rejected_state']
            by_arm[key[0]]['applied_operations'] += len(applied)
            by_arm[key[0]]['rejected_operations'] += len(rejected_parse) + len(rejected_state)
        if len(calls) != costs[key]['calls']:
            raise ValueError(f'model call trace count differs from costs for {key}')
        scoped = key[0] == 'scoped'
        call_rows = []
        event_kinds = {event['event_id']: event['kind'] for event in stream['events']}
        answer_calls = Counter()
        maintenance_calls = Counter()
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError(f'invalid model call trace row for {key}')
            generation = call.get('generation')
            purpose, event_id = call.get('purpose'), call.get('event_id')
            if (purpose not in PURPOSES or not isinstance(event_id, str) or event_id not in event_kinds
                    or not isinstance(call.get('messages'), list)
                    or not isinstance(call.get('raw_output'), str) or not isinstance(generation, dict)
                    or any(type(generation.get(field)) is not int or generation[field] < 0
                           for field in ('input_tokens', 'output_tokens'))
                    or type(call.get('service_seconds')) not in (int, float)
                    or not math.isfinite(call['service_seconds']) or call['service_seconds'] < 0):
                raise ValueError(f'invalid model call trace row for {key}')
            event_kind = event_kinds[event_id]
            if scoped:
                by_arm[key[0]][f'{purpose}_calls'] += 1
                call_rows.append({field: call.get(field) for field in (*CALL_FIELDS, *SCOPED_BATCH_FIELDS)})
                continue
            if purpose == 'answer':
                if event_kind != 'query':
                    raise ValueError(f'answer call must target a query event for {key}')
                answer_calls[event_id] += 1
                prediction = indexed_predictions[key[0]][(key[1], event_id)]
                if call['raw_output'] != prediction['answer'] or call['messages'] != prediction['messages']:
                    raise ValueError(f'answer call trace differs from prediction for {(key[0], key[1], event_id)}')
            elif purpose == 'maintenance':
                if key[0] != 'structured' or event_kind != 'observation':
                    raise ValueError(f'maintenance call must target a state-arm observation for {key}')
                maintenance_calls[event_id] += 1
            elif key[0] != 'structured' or event_kind != 'query':
                raise ValueError(f'link call must target a query event for {key}')
            call_rows.append({field: call[field] for field in CALL_FIELDS})
            by_arm[key[0]][f"{call['purpose']}_calls"] += 1
        calls_by_event = defaultdict(list)
        for call in calls:
            calls_by_event[call['event_id']].append(call)
        query_ids = [event['event_id'] for event in stream['events'] if event['kind'] == 'query']
        for query_id in query_ids:
            prediction = indexed_predictions[key[0]][(key[1], query_id)]
            required_seconds = math.fsum(
                call['service_seconds'] for call in calls_by_event[query_id]
            )
            if prediction['query_seconds'] < required_seconds and not math.isclose(
                    prediction['query_seconds'], required_seconds, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f'query_seconds is shorter than its attributed model service for '
                                 f'{(key[0], key[1], query_id)}')
        if scoped:
            _validate_scoped_batches(stream, ingress, calls, key)
        else:
            query_ids = [event['event_id'] for event in stream['events'] if event['kind'] == 'query']
            if any(answer_calls[event_id] != 1 for event_id in query_ids):
                raise ValueError(f'each query must have exactly one answer call for {key}')
            if key[0] == 'structured' and any(maintenance_calls[event_id] != 1 for event_id in observation_ids):
                raise ValueError(f'each structured observation must have one maintenance call for {key}')
        hashes[key] = hashlib.sha256(json.dumps(call_rows, ensure_ascii=False, sort_keys=True,
                                                separators=(',', ':')).encode('utf-8')).hexdigest()
        indexed[key] = trace
        by_arm[key[0]]['observations'] += len(ingress)
        by_arm[key[0]]['calls'] += len(calls)
    if indexed.keys() != expected:
        raise ValueError(f'missing ingress traces {sorted(expected - indexed.keys())}')
    for key, trace in indexed.items():
        purposes = Counter(call['purpose'] for call in trace['calls'])
        for purpose, fields in PURPOSE_COST_FIELDS.items():
            count_field, input_field, output_field, seconds_field = fields
            if count_field in costs[key] and costs[key][count_field] != purposes[purpose]:
                raise ValueError(f'{count_field} differs from model-call trace for {key}')
            for field, token in ((input_field, 'input_tokens'), (output_field, 'output_tokens')):
                actual = sum(call['generation'][token] for call in trace['calls']
                             if call['purpose'] == purpose)
                if field in costs[key] and costs[key][field] != actual:
                    raise ValueError(f'{field} differs from model-call trace for {key}')
            actual_seconds = math.fsum(call['service_seconds'] for call in trace['calls']
                                       if call['purpose'] == purpose)
            if seconds_field in costs[key] and not math.isclose(
                    costs[key][seconds_field], actual_seconds, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f'{seconds_field} differs from model-call trace for {key}')
        for field in ('input_tokens', 'output_tokens'):
            actual = sum(call['generation'][field] for call in trace['calls'])
            if costs[key][field] != actual:
                raise ValueError(f'{field} differs from model-call trace for {key}')
        actual_seconds = math.fsum(call['service_seconds'] for call in trace['calls'])
        if not math.isclose(costs[key]['model_seconds'], actual_seconds, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f'model_seconds differs from model-call trace for {key}')
    report = {'by_arm': by_arm, 'valid_stream_traces': len(indexed), 'model_call_sha256': [
        {'arm': arm, 'stream_id': stream_id, 'sha256': value}
        for (arm, stream_id), value in hashes.items()
    ]}
    return report, {'call_hashes': hashes, 'indexed': indexed}


def _read_run(path, streams, references, fixture_hash):
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get('format') != PREDICTION_FORMAT:
        raise ValueError('unsupported alias online v2 prediction format')
    if data.get('fixture_sha256') != fixture_hash:
        raise ValueError('prediction fixture_sha256 does not match fixture bytes')
    arms = data.get('arms')
    if arms not in (list(ARMS), list(reversed(ARMS))):
        raise ValueError(f'v2 arms must be {ARMS} or reversed')
    indexed = _index_predictions(data, references)
    costs = _index_costs(data, arms, streams)
    items = _score_items(streams, data, references, indexed)
    by_arm, diagnostics, resources = _aggregate(streams, items, costs, arms)
    traces, trace_lookup = _validate_traces(data, streams, costs, arms, indexed)
    split_results = {}
    for split in dict.fromkeys(stream.get('split', 'unspecified') for stream in streams):
        subset = [stream for stream in streams if stream.get('split', 'unspecified') == split]
        ids = {stream['stream_id'] for stream in subset}
        subset_items = [item for item in items if item['stream_id'] in ids]
        sub_by_arm, sub_diag, sub_resources = _aggregate(
            subset, subset_items, costs, arms,
        )
        split_results[split] = {'answers': sub_by_arm, 'diagnostics': sub_diag, 'resources': sub_resources,
                                'state_pair': _state_pair(subset_items),
                                'query_latency': _query_latency(indexed, arms, ids)}
    return {
        'prediction_sha256': hashlib.sha256(raw).hexdigest(),
        'arms': arms, 'checkpoint_files_sha256': data.get('checkpoint_files_sha256'),
        'config': data.get('config'), 'answer_summary': {
            'by_arm': by_arm, 'paired': [pair_results(items, left, right) for left, right in combinations(arms, 2)],
        },
        'diagnostics': diagnostics, 'state_pair': _state_pair(items), 'resources': resources, 'traces': traces,
        'by_split': split_results, 'query_latency': _query_latency(indexed, arms), 'items': items,
    }, indexed, trace_lookup


def compare_runs(main, repeat, main_index, repeat_index, main_traces, repeat_traces):
    if main['arms'] != list(reversed(repeat['arms'])):
        raise ValueError('repeat run does not reverse arm order')
    if main['checkpoint_files_sha256'] != repeat['checkpoint_files_sha256']:
        raise ValueError('checkpoint file hashes differ across repeat runs')
    if main.get('config') != repeat.get('config'):
        first, second = dict(main.get('config') or {}), dict(repeat.get('config') or {})
        first.pop('reverse_arms', None)
        second.pop('reverse_arms', None)
        if first != second:
            raise ValueError('inference config differs across repeat runs')
    per_arm = {}
    for arm in main['arms']:
        differences = []
        by_field = Counter()
        for key, left in main_index[arm].items():
            right = repeat_index[arm][key]
            changed = [field for field in REPRODUCIBLE_FIELDS if left.get(field) != right.get(field)]
            for field in changed:
                by_field[field] += 1
            if changed:
                differences.append({'stream_id': key[0], 'event_id': key[1], 'fields': changed})
        call_differences = []
        ingress_differences = []
        for stream_id in sorted(key[1] for key in main_traces['call_hashes'] if key[0] == arm):
            if main_traces['call_hashes'][arm, stream_id] != repeat_traces['call_hashes'][arm, stream_id]:
                call_differences.append(stream_id)
            left = main_traces['indexed'][arm, stream_id].get('ingress')
            right = repeat_traces['indexed'][arm, stream_id].get('ingress')
            if left != right:
                ingress_differences.append(stream_id)
        per_arm[arm] = {
            'query_items': len(main_index[arm]), 'identical': not differences and not call_differences
                         and not ingress_differences,
            'prediction_differences_by_field': dict(by_field), 'prediction_differences': differences,
            'model_call_stream_disagreements': call_differences,
            'ingress_stream_disagreements': ingress_differences,
            'primary_costs': main['resources'][arm], 'repeat_costs': repeat['resources'][arm],
        }
    return {'reverse_arm_order': True, 'by_arm': per_arm,
            'identical_all': all(item['identical'] for item in per_arm.values())}


def score_alias_online_v2(fixtures, predictions, repeat=None):
    fixture_bytes = fixtures.read_bytes()
    fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
    streams = load_streams(fixtures)
    if hashlib.sha256(fixtures.read_bytes()).hexdigest() != fixture_hash:
        raise ValueError('fixture changed while loading')
    references = query_references(streams)
    primary, primary_rows, primary_trace_info = _read_run(predictions, streams, references, fixture_hash)
    result = {'format': SCORE_FORMAT, 'fixture_sha256': fixture_hash, 'main': primary}
    if repeat is not None:
        repeated, repeated_rows, repeated_trace_info = _read_run(repeat, streams, references, fixture_hash)
        result['repeat'] = repeated
        result['rerun'] = compare_runs(primary, repeated, primary_rows, repeated_rows,
                                       primary_trace_info, repeated_trace_info)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--repeat', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score_alias_online_v2(args.fixtures, args.predictions, args.repeat)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
