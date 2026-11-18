import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.alias_online_compare_v2 import ARMS, compare
from evaluation.audit_alias_online_v2 import audit_alias_online_v2
from evaluation.continuous_event_protocol import public_stream


def _fixture_bytes():
    stream = {
        'format': 'pbslm.continuous_event_stream.v1',
        'stream_id': 'audit-synthetic',
        'split': 'holdout',
        'events': [
            {
                'event_id': 'o1', 'source_tick': 1, 'deliver_after_ms': 1,
                'kind': 'observation', 'text': '箱体A14放在东侧柜。',
            },
            {
                'event_id': 'q1', 'source_tick': 2, 'deliver_after_ms': 2,
                'kind': 'query', 'text': '箱体A14在哪里？',
                'oracle': {
                    'target': '不确定', 'entity': '箱体A14', 'entity_state': 'resolved',
                    'state': 'missing', 'case': 'audit', 'evidence_event_ids': ['o1'],
                },
            },
        ],
    }
    return (json.dumps(stream, ensure_ascii=False) + '\n').encode('utf-8')


def _generate(messages, *, max_new_tokens):
    output = '不确定' if max_new_tokens == 24 else '[]'
    metrics = {
        'input_tokens': 5,
        'output_tokens': len(output),
        'max_new_tokens': max_new_tokens,
        'finished_by_eos': True,
        'enable_thinking': False,
        'thinking_detected': False,
        'thinking_incomplete': False,
        'checkpoint_type': 'bfloat16',
        'device': 'cuda',
        'local_files_only': True,
    }
    return output, metrics


def _run(fixture_sha, reverse):
    stream_data = json.loads(_fixture_bytes())
    stream = public_stream(stream_data)
    arms = tuple(reversed(ARMS)) if reverse else ARMS
    result = compare(_generate, [stream], arms)
    return {
        'format': 'pbslm.alias_online_predictions.v2',
        'fixture_sha256': fixture_sha,
        'arms': list(arms),
        'checkpoint_files_sha256': {'weights': 'synthetic'},
        'provenance': {
            'git_commit': 'synthetic', 'git_status': '', 'source_sha256_at_capture': {},
            'python': 'synthetic', 'platform': 'synthetic', 'torch': 'synthetic',
        },
        'config': {'decode': 'synthetic', 'reverse_arms': reverse},
        **result,
    }


class AliasOnlineReplayAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.fixture_path = self.root / 'fixture.jsonl'
        fixture_bytes = _fixture_bytes()
        self.fixture_path.write_bytes(fixture_bytes)
        self.fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
        self.primary = self._write_run('primary.json', _run(self.fixture_hash, False))
        self.repeat = self._write_run('repeat.json', _run(self.fixture_hash, True))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_run(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        return path

    def _audit(self, primary=None, repeat=None):
        return audit_alias_online_v2(
            self.fixture_path, primary or self.primary, repeat or self.repeat,
        )

    def test_replays_all_arms_and_compares_reversed_runs(self):
        result = self._audit()
        self.assertEqual(result['primary']['verified_arm_stream_traces'], 4)
        self.assertEqual(result['primary']['predictions'], 4)
        self.assertTrue(result['comparison']['identical_predictions'])
        self.assertTrue(result['comparison']['identical_ingress'])
        self.assertTrue(result['comparison']['identical_calls'])
        self.assertEqual(result['primary']['per_arm_purpose_costs']['scoped']['maintenance']['calls'], 1)

    def test_rejects_ingress_operations_that_do_not_follow_raw_output(self):
        data = json.loads(self.primary.read_bytes())
        scoped = next(trace for trace in data['ingress'] if trace['arm'] == 'scoped')
        operation = ['set', '箱体A14', '东侧柜', None]
        scoped['ingress'][0]['validated_operations'] = [['register', '箱体A14', '', None], operation]
        scoped['ingress'][0]['applied_operations'] = [['register', '箱体A14', '', None], operation]
        altered = self._write_run('altered_ingress.json', data)
        with self.assertRaisesRegex(ValueError, 'ingress operations differ'):
            self._audit(primary=altered)

    def test_rejects_modified_raw_output_that_changes_replay(self):
        data = json.loads(self.primary.read_bytes())
        scoped = next(trace for trace in data['ingress'] if trace['arm'] == 'scoped')
        scoped['calls'][0]['raw_output'] = (
            '[ ["o1", "set", "箱体A14", "东侧柜", ""] ]'
        )
        altered = self._write_run('altered_output.json', data)
        with self.assertRaisesRegex(ValueError, 'predictions differ|ingress operations differ'):
            self._audit(primary=altered)

    def test_rejects_cost_totals_that_disagree_with_recorded_calls(self):
        data = json.loads(self.primary.read_bytes())
        scoped_cost = next(row for row in data['costs'] if row['arm'] == 'scoped')
        scoped_cost['input_tokens'] += 1
        altered = self._write_run('altered_cost.json', data)
        with self.assertRaisesRegex(ValueError, 'input_tokens differs'):
            self._audit(primary=altered)


if __name__ == '__main__':
    unittest.main()
