"""Audit frozen alias-online runs by replaying stored model outputs without inference."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from evaluation.alias_online_compare import replay
from evaluation.alias_online_compare_v2 import ARMS, replay_scoped
from evaluation.continuous_event_protocol import load_streams, public_stream


PREDICTION_FORMAT = 'pbslm.alias_online_predictions.v2'
PURPOSES = ('maintenance', 'link', 'answer')
TIMING_FIELDS = {'query_seconds', 'service_seconds'}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _without_timing(value):
    if isinstance(value, dict):
        return {key: _without_timing(item) for key, item in value.items() if key not in TIMING_FIELDS}
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    if isinstance(value, tuple):
        return [_without_timing(item) for item in value]
    return value


def _same(left, right):
    return _canonical(left) == _canonical(right)


def _expect(condition, message):
    if not condition:
        raise ValueError(message)


def _index_run(data, streams, fixture_hash, run_name):
    _expect(isinstance(data, dict) and data.get('format') == PREDICTION_FORMAT,
            f'{run_name}: unsupported prediction format')
    _expect(data.get('fixture_sha256') == fixture_hash, f'{run_name}: fixture hash mismatch')
    config = data.get('config')
    _expect(isinstance(config, dict), f'{run_name}: config must be an object')
    expected_arms = list(reversed(ARMS)) if config.get('reverse_arms') else list(ARMS)
    _expect(data.get('arms') == expected_arms, f'{run_name}: arm order disagrees with reverse_arms')

    stream_ids = set(streams)
    expected_keys = {(arm, stream_id) for arm in expected_arms for stream_id in stream_ids}
    traces = {}
    trace_rows = data.get('ingress')
    _expect(isinstance(trace_rows, list), f'{run_name}: ingress must be a list')
    for trace in trace_rows:
        _expect(isinstance(trace, dict), f'{run_name}: trace rows must be objects')
        key = trace.get('arm'), trace.get('stream_id')
        _expect(key in expected_keys and key not in traces, f'{run_name}: unexpected or duplicate trace {key}')
        _expect(isinstance(trace.get('calls'), list) and isinstance(trace.get('ingress'), list),
                f'{run_name}: missing calls or ingress for {key}')
        traces[key] = trace
    _expect(traces.keys() == expected_keys, f'{run_name}: incomplete stream traces')

    costs = {}
    cost_rows = data.get('costs')
    _expect(isinstance(cost_rows, list), f'{run_name}: costs must be a list')
    for row in cost_rows:
        _expect(isinstance(row, dict), f'{run_name}: cost rows must be objects')
        key = row.get('arm'), row.get('stream_id')
        _expect(key in expected_keys and key not in costs, f'{run_name}: unexpected or duplicate cost row {key}')
        costs[key] = row
    _expect(costs.keys() == expected_keys, f'{run_name}: incomplete cost rows')

    predictions = defaultdict(dict)
    expected_queries = {
        stream_id: {event['event_id'] for event in stream['events'] if event['kind'] == 'query'}
        for stream_id, stream in streams.items()
    }
    prediction_rows = data.get('predictions')
    _expect(isinstance(prediction_rows, list), f'{run_name}: predictions must be a list')
    for row in prediction_rows:
        _expect(isinstance(row, dict), f'{run_name}: prediction rows must be objects')
        key = row.get('arm'), row.get('stream_id')
        event_id = row.get('event_id')
        _expect(key in expected_keys and event_id in expected_queries[key[1]],
                f'{run_name}: unexpected prediction key {key + (event_id,)}')
        _expect(event_id not in predictions[key], f'{run_name}: duplicate prediction {key + (event_id,)}')
        predictions[key][event_id] = row
    for arm, stream_id in expected_keys:
        _expect(predictions[arm, stream_id].keys() == expected_queries[stream_id],
                f'{run_name}: incomplete predictions for {(arm, stream_id)}')
    return expected_arms, traces, costs, predictions


def _validate_costs(trace, cost, replay_cost, key):
    calls = trace['calls']
    for call in calls:
        _expect(isinstance(call, dict), f'model call rows must be objects for {key}')
        _expect(call.get('purpose') in PURPOSES and isinstance(call.get('generation'), dict),
                f'model call purpose and generation are required for {key}')
        generation = call['generation']
        _expect(all(type(generation.get(field)) is int and generation[field] >= 0
                    for field in ('input_tokens', 'output_tokens')),
                f'model call token counts are invalid for {key}')
        seconds = call.get('service_seconds')
        _expect(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
                f'model call service time is invalid for {key}')
    _expect(cost.get('calls') == len(calls), f'call count differs for {key}')
    for token_field in ('input_tokens', 'output_tokens'):
        actual = sum(call['generation'][token_field] for call in calls)
        _expect(cost.get(token_field) == actual, f'{token_field} differs from call trace for {key}')
    model_seconds = math.fsum(call['service_seconds'] for call in calls)
    _expect(math.isclose(cost['model_seconds'], model_seconds, rel_tol=1e-9, abs_tol=1e-9),
            f'model_seconds differs from call trace for {key}')
    _expect(math.isclose(cost['total_seconds'], cost['model_seconds'] + cost['host_seconds'],
                         rel_tol=1e-9, abs_tol=1e-9), f'total time does not reconcile for {key}')

    for purpose in PURPOSES:
        purpose_calls = [call for call in calls if call['purpose'] == purpose]
        expected = {
            f'{purpose}_calls': len(purpose_calls),
            f'{purpose}_input_tokens': sum(call['generation']['input_tokens'] for call in purpose_calls),
            f'{purpose}_output_tokens': sum(call['generation']['output_tokens'] for call in purpose_calls),
            f'{purpose}_model_seconds': math.fsum(call['service_seconds'] for call in purpose_calls),
        }
        count_alias = {'link': 'linker_calls'}.get(purpose, f'{purpose}_calls')
        if count_alias in cost:
            _expect(cost[count_alias] == expected[f'{purpose}_calls'],
                    f'{count_alias} differs from call trace for {key}')
        for field in (f'{purpose}_input_tokens', f'{purpose}_output_tokens', f'{purpose}_model_seconds'):
            if field in cost:
                actual = cost[field]
                target = expected[field]
                if field.endswith('_seconds'):
                    _expect(math.isclose(actual, target, rel_tol=1e-9, abs_tol=1e-9),
                            f'{field} differs from call trace for {key}')
                else:
                    _expect(actual == target, f'{field} differs from call trace for {key}')

    for field in ('peak_state_bytes', 'fallback_queries'):
        if field in cost and field in replay_cost:
            _expect(cost[field] == replay_cost[field], f'{field} differs from deterministic replay for {key}')


def _replay_stream_arm(stream, arm, recorded_trace, recorded_cost, predictions, key):
    calls = iter(recorded_trace['calls'])

    def recorded_generate(messages, *, max_new_tokens):
        call = next(calls, None)
        _expect(call is not None, f'replay made an unrecorded model call for {key}')
        _expect(_same(call.get('messages'), messages), f'model prompt differs during replay for {key}')
        _expect(call.get('generation', {}).get('max_new_tokens') == max_new_tokens,
                f'model token budget differs during replay for {key}')
        return call['raw_output'], call['generation']

    if arm == 'scoped':
        replay_predictions, replay_cost, replay_trace = replay_scoped(recorded_generate, stream)
    else:
        replay_predictions, replay_cost, replay_trace = replay(
            recorded_generate, stream, arm, record_query_seconds=True,
        )
    _expect(next(calls, None) is None, f'replay did not consume every model call for {key}')

    expected_predictions = [predictions[event['event_id']] for event in stream['events']
                            if event['kind'] == 'query']
    _expect(_same(_without_timing(replay_predictions), _without_timing(expected_predictions)),
            f'predictions differ from deterministic replay for {key}')
    _expect(_same(replay_trace['ingress'], recorded_trace['ingress']),
            f'ingress operations differ from raw-output replay for {key}')
    _expect(_same(_without_timing(replay_trace['calls']), _without_timing(recorded_trace['calls'])),
            f'call trace differs from deterministic replay for {key}')
    _validate_costs(recorded_trace, recorded_cost, replay_cost, key)
    return replay_predictions, replay_cost


def _purpose_totals(traces, arm):
    totals = {purpose: {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'model_seconds': 0.0}
              for purpose in PURPOSES}
    for trace in traces:
        if trace['arm'] != arm:
            continue
        for call in trace['calls']:
            row = totals[call['purpose']]
            row['calls'] += 1
            row['input_tokens'] += call['generation']['input_tokens']
            row['output_tokens'] += call['generation']['output_tokens']
            row['model_seconds'] = math.fsum((row['model_seconds'], call['service_seconds']))
    return totals


def _audit_run(raw, streams, fixture_hash, run_name):
    data = json.loads(raw)
    arms, traces, costs, predictions = _index_run(data, streams, fixture_hash, run_name)
    for stream_id, stream in streams.items():
        for arm in arms:
            key = arm, stream_id
            _replay_stream_arm(stream, arm, traces[key], costs[key], predictions[key], key)
    return {
        'name': run_name,
        'sha256': _sha256(raw),
        'fixture_sha256': fixture_hash,
        'arms': arms,
        'streams': len(streams),
        'predictions': len(data['predictions']),
        'verified_arm_stream_traces': len(traces),
        'per_arm_purpose_costs': {arm: _purpose_totals(data['ingress'], arm) for arm in arms},
        'data': data,
        'predictions_by_key': predictions,
        'traces_by_key': traces,
    }


def _compare_runs(primary, repeat):
    first, second = primary['data'], repeat['data']
    _expect(primary['arms'] == list(reversed(repeat['arms'])),
            'repeat run does not reverse the arm order')
    _expect(first.get('checkpoint_files_sha256') == second.get('checkpoint_files_sha256'),
            'checkpoint hashes differ between runs')

    first_config = dict(first.get('config') or {})
    second_config = dict(second.get('config') or {})
    first_config.pop('reverse_arms', None)
    second_config.pop('reverse_arms', None)
    _expect(_same(first_config, second_config), 'inference config differs between runs')

    first_provenance = dict(first.get('provenance') or {})
    second_provenance = dict(second.get('provenance') or {})
    first_provenance.pop('git_status', None)
    second_provenance.pop('git_status', None)
    _expect(_same(first_provenance, second_provenance), 'provenance differs between runs')

    prediction_differences = []
    for arm in primary['arms']:
        for stream_id in primary['traces_by_key']:
            if stream_id[0] != arm:
                continue
            left = primary['predictions_by_key'][stream_id]
            right = repeat['predictions_by_key'][arm, stream_id[1]]
            for event_id in left:
                if not _same(_without_timing(left[event_id]), _without_timing(right[event_id])):
                    prediction_differences.append([arm, stream_id[1], event_id])

    ingress_differences, call_differences = [], []
    for key, left in primary['traces_by_key'].items():
        right = repeat['traces_by_key'][key]
        if not _same(left['ingress'], right['ingress']):
            ingress_differences.append(list(key))
        if not _same(_without_timing(left['calls']), _without_timing(right['calls'])):
            call_differences.append(list(key))
    return {
        'reverse_arm_order': True,
        'checkpoint_hashes_equal': True,
        'config_equal_except_reverse_flag': True,
        'provenance_equal_except_git_status': True,
        'identical_predictions': not prediction_differences,
        'identical_ingress': not ingress_differences,
        'identical_calls': not call_differences,
        'prediction_differences': prediction_differences,
        'ingress_differences': ingress_differences,
        'call_differences': call_differences,
    }


def _verify_manifest(manifest_path, fixture_hash, runs, repo_root):
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    _expect(manifest.get('format') == 'pbslm.alias_online_freeze.v2', 'unsupported freeze manifest')
    expected_fixture = manifest['fixture_sha256']['holdout']['sha256']
    _expect(fixture_hash == expected_fixture, 'fixture hash differs from freeze manifest')
    expected_checkpoint = {name: value['sha256']
                          for name, value in manifest['checkpoint']['checkpoint_files'].items()}
    for run in runs:
        data = run['data']
        provenance = data.get('provenance') or {}
        _expect(data.get('checkpoint_files_sha256') == expected_checkpoint,
                f"{run['name']}: checkpoint differs from freeze manifest")
        _expect(provenance.get('git_commit') == manifest['git']['base_commit'],
                f"{run['name']}: base commit differs from freeze manifest")
        _expect(provenance.get('source_sha256_at_capture') == manifest['source_sha256'],
                f"{run['name']}: source hashes differ from freeze manifest")
        _expect(_same({key: provenance.get(key) for key in manifest['runtime']}, manifest['runtime']),
                f"{run['name']}: runtime differs from freeze manifest")

    hashed_groups = (
        ('source_sha256', manifest['source_sha256']),
        ('test_sha256', manifest.get('test_sha256', {})),
        ('repo_controls_sha256', manifest.get('repo_controls_sha256', {})),
        ('scoring_dependencies_sha256', manifest.get('scoring_dependencies_sha256', {})),
    )
    checked = {}
    for group_name, files in hashed_groups:
        for name, expected in files.items():
            path = repo_root / name
            _expect(path.is_file(), f'manifest file is missing: {name}')
            actual = _sha256(path.read_bytes())
            _expect(actual == expected, f'manifest file hash differs: {name}')
            checked[name] = actual
    protocol_path = repo_root / 'experiments/continuous_online/ALIAS_SCOPED_V2_PROTOCOL.md'
    protocol_hash = _sha256(protocol_path.read_bytes())
    _expect(protocol_hash == manifest['protocol_sha256'], 'protocol hash differs from freeze manifest')
    return {
        'sha256': _sha256(raw),
        'verified_frozen_files': len(checked) + 1,
        'checkpoint_files': len(expected_checkpoint),
        'runtime_matches': True,
        'source_hashes_match': True,
    }


def audit_alias_online_v2(fixtures, primary_path, repeat_path, *, freeze_manifest=None, repo_root=None):
    fixture_bytes = fixtures.read_bytes()
    fixture_hash = _sha256(fixture_bytes)
    streams = {stream['stream_id']: public_stream(stream) for stream in load_streams(fixtures)}
    _expect(_sha256(fixtures.read_bytes()) == fixture_hash, 'fixture changed while loading')

    primary = _audit_run(primary_path.read_bytes(), streams, fixture_hash, primary_path.name)
    repeat = _audit_run(repeat_path.read_bytes(), streams, fixture_hash, repeat_path.name)
    comparison = _compare_runs(primary, repeat)
    result = {
        'format': 'pbslm.alias_online_replay_audit.v1',
        'fixture_sha256': fixture_hash,
        'primary': {key: value for key, value in primary.items()
                    if key not in {'data', 'predictions_by_key', 'traces_by_key'}},
        'repeat': {key: value for key, value in repeat.items()
                   if key not in {'data', 'predictions_by_key', 'traces_by_key'}},
        'comparison': comparison,
    }
    if freeze_manifest is not None:
        root = (repo_root or Path.cwd()).resolve()
        result['freeze_manifest'] = _verify_manifest(
            freeze_manifest, fixture_hash, (primary, repeat), root,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--repeat', type=Path, required=True)
    parser.add_argument('--freeze-manifest', type=Path)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        result = audit_alias_online_v2(
            args.fixtures, args.primary, args.repeat,
            freeze_manifest=args.freeze_manifest, repo_root=args.repo_root,
        )
    except (OSError, KeyError, StopIteration, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
