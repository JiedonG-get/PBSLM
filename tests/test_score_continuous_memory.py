import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from evaluation.continuous_event_protocol import FORMAT
from evaluation.score_continuous_memory import UNKNOWN, score, score_files


def stream_fixture():
    return {'format': FORMAT, 'stream_id': 'stream-1', 'events': [
        {'event_id': 'o1', 'source_tick': 1, 'deliver_after_ms': 0,
         'kind': 'observation', 'text': 'The item moved to Cabinet A.'},
        {'event_id': 'q1', 'source_tick': 2, 'deliver_after_ms': 10,
         'kind': 'query', 'text': 'Where is the item?', 'oracle': {'target': 'Cabinet A'}},
        {'event_id': 'q2', 'source_tick': 3, 'deliver_after_ms': 20,
         'kind': 'query', 'text': 'Who moved it?', 'oracle': {'target': UNKNOWN, 'case': 'unknown'}},
    ]}


def predictions_for(*answers):
    arms = ['history', 'state'][:len(answers)]
    return {'arms': arms, 'predictions': [
        {'arm': arm, 'stream_id': 'stream-1', 'event_id': event_id,
         'answer': answer, 'unused_metric': 7}
        for arm, arm_answers in zip(arms, answers)
        for event_id, answer in zip(('q1', 'q2'), arm_answers)
    ]}


class ContinuousMemoryScorerTests(unittest.TestCase):
    def test_normalization_coverage_and_disagreements(self):
        data = predictions_for((' Cabinet \uff21\u3002 ', UNKNOWN + '\uff01'),
                               ('It is in Cabinet A', '\u3000'))
        result = score([stream_fixture()], data)

        history = result['by_arm']['history']
        self.assertEqual(history['overall']['accuracy'], 1.0)
        self.assertEqual(history['known']['coverage'], 1.0)
        self.assertEqual(history['unknown']['accuracy'], 1.0)
        self.assertEqual(history['unknown']['coverage'], 0.0)
        self.assertEqual(history['unknown']['abstained'], 1)
        self.assertEqual(history['by_case']['unspecified']['items'], 1)
        self.assertEqual(history['by_case']['unknown']['items'], 1)
        self.assertEqual(result['by_arm']['state']['overall']['accuracy'], 0.0)
        self.assertEqual(result['by_arm']['state']['unknown']['blank'], 1)
        self.assertEqual(result['items'][0]['outcomes']['history']['normalized_answer'], 'Cabinet A')
        pair, = result['paired']
        self.assertEqual(pair['arms'], ['history', 'state'])
        self.assertEqual(pair['left_only_correct'], 2)
        self.assertEqual(pair['answer_disagreements'], 2)
        self.assertEqual([item['event_id'] for item in pair['disagreements']], ['q1', 'q2'])

    def test_rejects_duplicate_unknown_and_missing_predictions(self):
        for problem in ('duplicate', 'unknown', 'missing'):
            with self.subTest(problem=problem):
                data = predictions_for(('Cabinet A', UNKNOWN))
                rows = data['predictions']
                if problem == 'duplicate':
                    rows.append(dict(rows[0]))
                elif problem == 'unknown':
                    rows[0]['event_id'] = 'o1'
                else:
                    rows.pop()
                with self.assertRaisesRegex(ValueError, problem):
                    score([stream_fixture()], data)

    def test_sealed_artifact_hashes(self):
        with TemporaryDirectory() as directory:
            fixture = Path(directory) / 'fixtures.jsonl'
            fixture_bytes = (json.dumps(stream_fixture()) + '\n').encode('utf-8')
            fixture.write_bytes(fixture_bytes)
            fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
            predictions = Path(directory) / 'predictions.json'
            data = predictions_for(('Cabinet A', UNKNOWN))
            data['fixture_sha256'] = fixture_hash
            prediction_bytes = json.dumps(data).encode('utf-8')
            predictions.write_bytes(prediction_bytes)

            result = score_files(fixture, predictions)
            self.assertEqual(result['fixture_sha256'], fixture_hash)
            self.assertEqual(result['prediction_sha256'], hashlib.sha256(prediction_bytes).hexdigest())
            output = Path(directory) / 'scores' / 'result.json'
            subprocess.run(
                [sys.executable, '-B', '-m', 'evaluation.score_continuous_memory',
                 '--fixtures', str(fixture), '--predictions', str(predictions), '--output', str(output)],
                check=True, capture_output=True,
            )
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')), result)
            fixture.write_bytes(fixture_bytes + b'\n')
            with self.assertRaisesRegex(ValueError, 'fixture_sha256'):
                score_files(fixture, predictions)


if __name__ == '__main__':
    unittest.main()
