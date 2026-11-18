import unittest

from evaluation.continuous_event_protocol import FORMAT, public_stream, validate
from evaluation.continuous_memory_compare import compare


def fixture():
    events = []
    for i, (kind, text) in enumerate((('observation', '门口钥匙由小吴保管。'),
                                      ('observation', '窗户已关好。'),
                                      ('query', '谁保管门口钥匙？'),
                                      ('observation', '后来换成小田保管。')), 1):
        events.append({'event_id': str(i), 'source_tick': i, 'deliver_after_ms': i,
                       'kind': kind, 'text': text, 'oracle': {'target': 'PRIVATE LABEL'}})
    return {'format': FORMAT, 'stream_id': 'stream', 'events': events}


class ContinuousMemoryCompareTests(unittest.TestCase):
    def test_arrived_text_only_and_distinct_evidence_windows(self):
        def generate(messages, *, max_new_tokens):
            content = messages[-1]['content']
            self.assertNotIn('PRIVATE LABEL', content)
            return '保存中' if '新观察' in content else '小吴', {
                'input_tokens': 10, 'output_tokens': 2, 'finished_by_eos': True,
            }

        report = compare(generate, [public_stream(validate(fixture()))], window=1,
                         state_chars=2, query_tokens=4, update_tokens=8)
        rows = {row['arm']: row for row in report['predictions']}
        self.assertEqual(rows['history']['evidence_event_ids'], ['1', '2'])
        self.assertEqual(rows['recent']['evidence_event_ids'], ['2'])
        self.assertEqual(rows['retrieval']['evidence_event_ids'], ['1'])
        self.assertEqual(rows['retrieval']['retrieval_rows_scanned'], 2)
        self.assertEqual(rows['bounded_summary']['evidence_event_ids'], [])
        self.assertEqual(rows['bounded_summary']['evidence_chars'], 2)
        self.assertTrue(all('小田' not in call['messages'][-1]['content']
                            for call in report['calls'] if call['purpose'] == 'query'))
        self.assertEqual(len(report['state_updates'][0]['updates']), 3)

    def test_fixture_rejects_backward_schedule(self):
        data = fixture()
        data['events'][1]['deliver_after_ms'] = 0
        with self.assertRaisesRegex(ValueError, 'nondecreasing'):
            validate(data)
