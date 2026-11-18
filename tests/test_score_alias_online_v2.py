import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.alias_online_compare_v2 import EXTRACT_PROMPT as SCOPED_EXTRACT_PROMPT
from evaluation.score_alias_online_v2 import ARMS, score_alias_online_v2


def _fixture():
    events = []
    scenarios = (
        ('located', 'resolved', 'located', '工具箱A14', '东侧柜', ['o1']),
        ('no_location', 'resolved', 'no_location', '工具箱B22', '不确定', ['o2']),
        ('missing', 'resolved', 'missing', '工具箱C33', '不确定', ['o3']),
        ('unknown', 'unknown', 'missing', None, '不确定', []),
        ('ambiguous', 'ambiguous', 'missing', None, '不确定', []),
    )
    observations = []
    queries = []
    for index, (case, entity_state, state, entity, target, evidence) in enumerate(scenarios, start=1):
        observation_id = f'o{index}'
        query_id = f'q{index}'
        observations.append(observation_id)
        events.append({
            'event_id': observation_id, 'source_tick': 2 * index - 1,
            'deliver_after_ms': 2 * index - 1, 'kind': 'observation',
            'text': f'观察记录 {index}',
        })
        events.append({
            'event_id': query_id, 'source_tick': 2 * index,
            'deliver_after_ms': 2 * index, 'kind': 'query', 'text': f'查询 {index}',
            'oracle': {
                'target': target, 'entity': entity, 'entity_state': entity_state,
                'state': state, 'case': case, 'evidence_event_ids': evidence,
            },
        })
        queries.append((query_id, entity_state, state, entity, target, evidence, case))
    observations.append('trailing')
    events.append({
        'event_id': 'trailing', 'source_tick': 11, 'deliver_after_ms': 11,
        'kind': 'observation', 'text': '尾部观察记录',
    })
    stream = {'format': 'pbslm.continuous_event_stream.v1', 'stream_id': 'synthetic',
              'split': 'dev', 'events': events}
    return stream, observations, queries


def _prediction_data(fixture_sha256, arm_order=None):
    stream, observations, queries = _fixture()
    arms = list(arm_order or ARMS)
    predictions, costs, traces = [], [], []
    for arm in arms:
        for query_id, entity_state, state, entity, target, evidence, case in queries:
            predicted_entity = entity
            predicted_entity_status = entity_state
            if arm == 'structured' and query_id == 'q1':
                predicted_entity = '工具箱A99'
            if arm == 'scoped' and case == 'unknown':
                predicted_entity_status = 'ambiguous'
            row = {
                'arm': arm, 'stream_id': 'synthetic', 'event_id': query_id,
                'answer': target, 'evidence_event_ids': list(evidence),
                'messages': [], 'available_observations': len(observations),
                'query_seconds': 1.0 + int(query_id[1:]) * 0.2
                                 + {'history': 0.0, 'lexical': 0.1, 'structured': 0.2, 'scoped': 0.3}[arm],
            }
            if arm in {'structured', 'scoped'}:
                row.update({
                    'entity': predicted_entity,
                    'readout': '东侧柜' if state == 'located' else None,
                    'link_method': 'direct',
                    'memory_evidence_event_ids': list(evidence),
                })
            if arm == 'scoped':
                row.update({
                    'entity_status': predicted_entity_status,
                    'location_status': state if entity is not None else None,
                    'fallback': False,
                })
            predictions.append(row)

        answers_by_event = {
            row['event_id']: row['answer'] for row in predictions if row['arm'] == arm
        }
        ingress = []
        calls = []
        token_totals = {'input_tokens': 0, 'output_tokens': 0, 'model_seconds': 0.0}
        purpose_totals = {
            purpose: {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'model_seconds': 0.0}
            for purpose in ('maintenance', 'link', 'answer')
        }
        pending = []
        for event in stream['events']:
            event_id = event['event_id']
            batch_fields = {}
            if event['kind'] == 'observation' and arm in {'structured', 'scoped'}:
                row = {
                    'event_id': event_id, 'validated_operations': [],
                    'applied_operations': [], 'rejected_parse': [], 'rejected_state': [],
                }
                ingress.append(row)
                if arm == 'scoped':
                    row['maintenance_skipped'] = event_id in {'o4', 'trailing'}
                    if not row['maintenance_skipped']:
                        pending.append(event_id)
                    purposes = ()
                else:
                    purposes = ('maintenance',)
            elif event['kind'] == 'query':
                if arm == 'scoped':
                    purposes = ('maintenance',) if pending else ()
                    batch_fields = {'query_event_id': event_id, 'observation_event_ids': pending}
                    pending = []
                else:
                    purposes = ('link', 'answer') if arm == 'structured' else ('answer',)
            else:
                purposes = ()
            for purpose in purposes:
                generation = {'input_tokens': 3, 'output_tokens': 2}
                service_seconds = 0.125
                call_messages = []
                if arm == 'scoped' and purpose == 'maintenance':
                    observation_by_id = {event['event_id']: event for event in stream['events']
                                         if event['kind'] == 'observation'}
                    batch = [{'event_id': observation_id, 'text': observation_by_id[observation_id]['text']}
                             for observation_id in batch_fields['observation_event_ids']]
                    call_messages = [
                        {'role': 'system', 'content': SCOPED_EXTRACT_PROMPT},
                        {'role': 'user', 'content': json.dumps(batch, ensure_ascii=False)},
                    ]
                calls.append({
                    'event_id': event_id, 'purpose': purpose, 'messages': call_messages,
                    'raw_output': answers_by_event[event_id] if purpose == 'answer' else 'synthetic',
                    'generation': generation,
                    'service_seconds': service_seconds,
                    **batch_fields,
                })
                totals = purpose_totals[purpose]
                totals['calls'] += 1
                totals['input_tokens'] += generation['input_tokens']
                totals['output_tokens'] += generation['output_tokens']
                totals['model_seconds'] += service_seconds
                token_totals['input_tokens'] += generation['input_tokens']
                token_totals['output_tokens'] += generation['output_tokens']
                token_totals['model_seconds'] += service_seconds

        traces.append({'arm': arm, 'stream_id': 'synthetic', 'ingress': ingress, 'calls': calls})
        costs.append({
            'arm': arm, 'stream_id': 'synthetic', 'calls': len(calls),
            **token_totals,
            'total_seconds': token_totals['model_seconds'],
            'host_seconds': 0.0,
            'maintenance_calls': purpose_totals['maintenance']['calls'],
            'maintenance_input_tokens': purpose_totals['maintenance']['input_tokens'],
            'maintenance_output_tokens': purpose_totals['maintenance']['output_tokens'],
            'maintenance_model_seconds': purpose_totals['maintenance']['model_seconds'],
            'linker_calls': purpose_totals['link']['calls'],
            'link_input_tokens': purpose_totals['link']['input_tokens'],
            'link_output_tokens': purpose_totals['link']['output_tokens'],
            'link_model_seconds': purpose_totals['link']['model_seconds'],
            'answer_calls': purpose_totals['answer']['calls'],
            'answer_input_tokens': purpose_totals['answer']['input_tokens'],
            'answer_output_tokens': purpose_totals['answer']['output_tokens'],
            'answer_model_seconds': purpose_totals['answer']['model_seconds'],
            'fallback_queries': 0, 'peak_state_bytes': 0,
        })
    return {
        'format': 'pbslm.alias_online_predictions.v2',
        'fixture_sha256': fixture_sha256,
        'arms': arms,
        'checkpoint_files_sha256': {'weights': 'synthetic'},
        'config': {'decode': 'greedy', 'reverse_arms': arms[0] == 'scoped'},
        'predictions': predictions, 'costs': costs, 'ingress': traces,
    }


class AliasOnlineV2ScorerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.fixture_path = self.root / 'fixture.jsonl'
        stream, _, _ = _fixture()
        self.fixture_bytes = (json.dumps(stream, ensure_ascii=False) + '\n').encode('utf-8')
        self.fixture_path.write_bytes(self.fixture_bytes)
        self.fixture_sha256 = hashlib.sha256(self.fixture_bytes).hexdigest()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_predictions(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        return path

    def test_scores_entity_and_location_states_and_pairwise_outputs(self):
        primary = self._write_predictions('primary.json', _prediction_data(self.fixture_sha256))
        report = score_alias_online_v2(self.fixture_path, primary)

        scoped = report['main']['diagnostics']['scoped']['overall']
        self.assertEqual(scoped['entities']['by_gold_status']['resolved']['items'], 3)
        self.assertEqual(scoped['entities']['by_gold_status']['unknown']['items'], 1)
        self.assertEqual(scoped['entities']['by_gold_status']['ambiguous']['items'], 1)
        self.assertEqual(scoped['location_status']['by_gold']['located']['items'], 1)
        self.assertEqual(scoped['location_status']['by_gold']['no_location']['items'], 1)
        self.assertEqual(scoped['location_status']['by_gold']['missing']['items'], 1)
        self.assertEqual(scoped['location_status']['by_gold']['unresolved']['items'], 2)
        self.assertEqual(scoped['entities']['status_compatible'], 5)
        self.assertEqual(scoped['entities']['status_correct'], 4)
        structured_items = report['main']['items']
        self.assertFalse(structured_items[0]['outcomes']['structured']['entity_correct'])
        self.assertTrue(structured_items[0]['outcomes']['structured']['readout_correct'])
        self.assertFalse(structured_items[0]['outcomes']['structured']['entity_and_readout_correct'])
        self.assertIn('entity_and_readout_correct', report['main']['state_pair'])
        self.assertEqual(report['main']['traces']['by_arm']['history']['observations'], 0)
        self.assertEqual(report['main']['traces']['by_arm']['scoped']['observations'], 6)
        self.assertEqual(report['main']['resources']['scoped']['maintenance_calls'], 4)
        self.assertEqual(report['main']['resources']['scoped']['answer_calls'], 0)

    def test_rejects_future_oracle_and_prediction_evidence(self):
        stream, _, _ = _fixture()
        stream['events'][1]['oracle']['evidence_event_ids'] = ['o2']
        invalid_fixture = self.root / 'future-oracle.jsonl'
        invalid_bytes = (json.dumps(stream, ensure_ascii=False) + '\n').encode('utf-8')
        invalid_fixture.write_bytes(invalid_bytes)
        invalid_prediction = self._write_predictions(
            'future-oracle-prediction.json',
            _prediction_data(hashlib.sha256(invalid_bytes).hexdigest()),
        )
        with self.assertRaisesRegex(ValueError, 'prior observations'):
            score_alias_online_v2(invalid_fixture, invalid_prediction)

        bad_data = _prediction_data(self.fixture_sha256)
        row = next(row for row in bad_data['predictions']
                   if row['arm'] == 'structured' and row['event_id'] == 'q1')
        row['evidence_event_ids'] = ['o2']
        bad_prediction = self._write_predictions('future-prediction.json', bad_data)
        with self.assertRaisesRegex(ValueError, 'prior observations'):
            score_alias_online_v2(self.fixture_path, bad_prediction)

    def test_rejects_unresolved_scoped_readout(self):
        bad_data = _prediction_data(self.fixture_sha256)
        row = next(row for row in bad_data['predictions']
                   if row['arm'] == 'scoped' and row['event_id'] == 'q4')
        row['readout'] = '东侧柜'
        bad_prediction = self._write_predictions('bad-unresolved.json', bad_data)
        with self.assertRaisesRegex(ValueError, 'unresolved entity requires null'):
            score_alias_online_v2(self.fixture_path, bad_prediction)

    def test_rejects_costs_that_disagree_with_call_trace(self):
        bad_data = _prediction_data(self.fixture_sha256)
        bad_data['costs'][0]['answer_calls'] += 1
        bad_prediction = self._write_predictions('bad-costs.json', bad_data)
        with self.assertRaisesRegex(ValueError, 'answer_calls differs'):
            score_alias_online_v2(self.fixture_path, bad_prediction)

    def test_rejects_scoped_batches_with_future_observations(self):
        data = _prediction_data(self.fixture_sha256)
        trace = next(trace for trace in data['ingress'] if trace['arm'] == 'scoped')
        trace['calls'][0]['observation_event_ids'] = ['o2']
        prediction = self._write_predictions('future-batch.json', data)
        with self.assertRaisesRegex(ValueError, 'must match non-skipped ingress'):
            score_alias_online_v2(self.fixture_path, prediction)

    def test_rejects_non_skipped_trailing_scoped_observation(self):
        data = _prediction_data(self.fixture_sha256)
        trace = next(trace for trace in data['ingress'] if trace['arm'] == 'scoped')
        trace['ingress'][-1]['maintenance_skipped'] = False
        prediction = self._write_predictions('unprocessed-tail.json', data)
        with self.assertRaisesRegex(ValueError, 'trailing observations must be skipped'):
            score_alias_online_v2(self.fixture_path, prediction)

    def test_rejects_scoped_batches_with_shortened_observation_text(self):
        data = _prediction_data(self.fixture_sha256)
        trace = next(trace for trace in data['ingress'] if trace['arm'] == 'scoped')
        trace['calls'][0]['messages'][1]['content'] = json.dumps(
            [{'event_id': 'o1', 'text': '缩短后的原文'}], ensure_ascii=False,
        )
        prediction = self._write_predictions('shortened-batch.json', data)
        with self.assertRaisesRegex(ValueError, 'prompt differs from its declared fixture batch'):
            score_alias_online_v2(self.fixture_path, prediction)

    def test_rejects_query_seconds_below_attributed_model_service(self):
        data = _prediction_data(self.fixture_sha256)
        prediction = next(row for row in data['predictions']
                          if row['arm'] == 'scoped' and row['event_id'] == 'q1')
        prediction['query_seconds'] = 0.124
        path = self._write_predictions('short-query-time.json', data)
        with self.assertRaisesRegex(ValueError, 'shorter than its attributed model service'):
            score_alias_online_v2(self.fixture_path, path)

    def test_rejects_missing_or_negative_query_seconds(self):
        missing_data = _prediction_data(self.fixture_sha256)
        missing_data['predictions'][0].pop('query_seconds')
        missing = self._write_predictions('missing-query-time.json', missing_data)
        with self.assertRaisesRegex(ValueError, 'query_seconds must be finite and nonnegative'):
            score_alias_online_v2(self.fixture_path, missing)

        negative_data = _prediction_data(self.fixture_sha256)
        negative_data['predictions'][0]['query_seconds'] = -0.1
        negative = self._write_predictions('negative-query-time.json', negative_data)
        with self.assertRaisesRegex(ValueError, 'query_seconds must be finite and nonnegative'):
            score_alias_online_v2(self.fixture_path, negative)

    def test_reports_query_latency_distribution_per_arm(self):
        prediction = self._write_predictions('latency.json', _prediction_data(self.fixture_sha256))
        report = score_alias_online_v2(self.fixture_path, prediction)
        history = report['main']['query_latency']['history']
        self.assertEqual(history['queries'], 5)
        self.assertAlmostEqual(history['total_seconds'], 8.0)
        self.assertAlmostEqual(history['median_seconds'], 1.6)
        self.assertAlmostEqual(history['p90_seconds'], 2.0)
        self.assertEqual(history['min_seconds'], 1.2)
        self.assertEqual(history['max_seconds'], 2.0)
        self.assertEqual(report['main']['by_split']['dev']['query_latency']['history'], history)

    def test_rejects_scoped_answer_that_differs_from_readout(self):
        data = _prediction_data(self.fixture_sha256)
        prediction = next(row for row in data['predictions'] if row['arm'] == 'scoped')
        prediction['answer'] = '东侧柜。'
        path = self._write_predictions('bad-answer.json', data)
        with self.assertRaisesRegex(ValueError, 'scoped answer must match the direct state readout'):
            score_alias_online_v2(self.fixture_path, path)

    def test_repeat_reverses_arm_order_and_reports_prediction_differences(self):
        primary_data = _prediction_data(self.fixture_sha256)
        repeat_data = _prediction_data(self.fixture_sha256, list(reversed(ARMS)))
        changed = next(row for row in repeat_data['predictions'] if row['arm'] == 'history')
        changed['answer'] = '错误地点'
        answer_call = next(
            call for trace in repeat_data['ingress'] if trace['arm'] == changed['arm']
            for call in trace['calls'] if call['event_id'] == changed['event_id'] and call['purpose'] == 'answer'
        )
        answer_call['raw_output'] = changed['answer']
        primary = self._write_predictions('primary.json', primary_data)
        repeat = self._write_predictions('repeat.json', repeat_data)
        report = score_alias_online_v2(self.fixture_path, primary, repeat)
        self.assertTrue(report['rerun']['reverse_arm_order'])
        self.assertFalse(report['rerun']['identical_all'])
        self.assertEqual(
            report['rerun']['by_arm']['history']['prediction_differences_by_field']['answer'], 1,
        )


if __name__ == '__main__':
    unittest.main()
