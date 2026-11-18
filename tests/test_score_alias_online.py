import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.continuous_event_protocol import FORMAT
from evaluation.score_alias_online import ARMS, PREDICTION_FORMAT, STRUCTURED_ARMS, score_alias_snapshots
from evaluation.score_continuous_memory import UNKNOWN


def event(index, kind, text, **extra):
    return {'event_id': str(index), 'source_tick': index, 'deliver_after_ms': index,
            'kind': kind, 'text': text, **extra}


class AliasScoringTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.fixture = Path(folder.name) / 'fixture.jsonl'
        self.predictions = Path(folder.name) / 'predictions.json'
        self.repeat = Path(folder.name) / 'repeat.json'
        stream = {'format': FORMAT, 'stream_id': 'stream', 'split': 'dev', 'events': [
            event(1, 'observation', 'Item Q17 is in Room 3.'),
            event(2, 'observation', 'Q17 also goes by Blue.'),
            event(3, 'query', 'Where is Blue?', oracle={
                'target': 'Room 3', 'entity': 'Q17', 'case': 'alias', 'evidence_event_ids': ['1', '2'],
            }),
            event(4, 'query', 'Where is Green?', oracle={
                'target': UNKNOWN, 'entity': None, 'case': 'unknown', 'evidence_event_ids': [],
            }),
            event(5, 'observation', 'Room 4 opens at noon.'),
        ]}
        self.fixture.write_text(json.dumps(stream) + '\n', encoding='utf-8')
        self.report = {'format': PREDICTION_FORMAT, 'arms': list(ARMS),
                       'fixture_sha256': hashlib.sha256(self.fixture.read_bytes()).hexdigest(),
                       'predictions': [], 'costs': []}
        for arm in ARMS:
            for event_id in ('3', '4'):
                structured = arm in STRUCTURED_ARMS
                known = event_id == '3'
                fallback = arm == 'structured_fallback' and known
                answer = 'Room 3.' if (arm == 'history' or fallback) and known else UNKNOWN
                if not known and arm == 'structured':
                    answer = 'Room 2'
                evidence = ['1', '2'] if arm in ('history', 'structured_fallback') else ['2']
                self.report['predictions'].append({
                    'arm': arm, 'stream_id': 'stream', 'event_id': event_id, 'answer': answer,
                    'entity': 'Q17' if structured and known else None, 'readout': None,
                    'evidence_event_ids': evidence, 'messages': [], 'fallback': fallback,
                })
            self.report['costs'].append({'arm': arm, 'stream_id': 'stream', 'calls': 2,
                                         'input_tokens': 34, 'output_tokens': 8, 'model_seconds': .5,
                                         'maintenance_calls': 1, 'peak_state_bytes': 70})

    def save(self, path, data):
        path.write_text(json.dumps(data), encoding='utf-8')

    def test_outcomes_resources_and_reversed_order(self):
        self.save(self.predictions, self.report)
        self.save(self.repeat, {**self.report, 'arms': list(reversed(ARMS))})
        result = score_alias_snapshots(self.fixture, self.predictions, self.repeat)
        summary = result['main']['overall']
        diagnostics = summary['diagnostics']
        self.assertEqual(summary['answers']['by_arm']['history']['overall']['correct'], 2)
        self.assertEqual(summary['answers']['paired'][0]['left_only_correct'], 1)
        self.assertEqual(diagnostics['history']['overall']['evidence']['queries_with_gold_links'], 1)
        self.assertEqual(diagnostics['lexical']['overall']['evidence']['gold_links_found'], 1)
        self.assertEqual(diagnostics['lexical']['overall']['over_abstained'], 1)
        self.assertEqual(diagnostics['structured']['overall']['under_abstained'], 1)
        self.assertEqual(diagnostics['structured_fallback']['overall']['readout']['readout_wrong_answer_correct'], 1)
        self.assertEqual(diagnostics['structured_fallback']['overall']['fallback_answers']['items'], 1)
        self.assertEqual(diagnostics['structured']['overall']['entities']['correct'], 2)
        self.assertEqual(summary['resources']['structured']['maintenance_calls'], 1)
        self.assertEqual(summary['resources']['history']['peak_state_bytes'], 70)
        self.assertTrue(result['rerun']['reverse_arm_order'])
        self.assertTrue(result['rerun']['by_arm']['history']['identical'])

    def test_future_evidence_and_fixture_mismatch_are_rejected(self):
        self.report['predictions'][0]['evidence_event_ids'].append('5')
        self.save(self.predictions, self.report)
        with self.assertRaisesRegex(ValueError, 'prior observations'):
            score_alias_snapshots(self.fixture, self.predictions)
        self.report['fixture_sha256'] = 'invalid'
        self.save(self.predictions, self.report)
        with self.assertRaisesRegex(ValueError, 'fixture_sha256'):
            score_alias_snapshots(self.fixture, self.predictions)


if __name__ == '__main__':
    unittest.main()
