import unittest

from evaluation.online_story_contract import load_cases
from evaluation.online_story_trace import score_trace


class StoryTraceTests(unittest.TestCase):
    def setUp(self):
        self.case = load_cases()['cases'][0]

    def test_commit_is_scored_against_delivered_state(self):
        trace = [
            {'kind': 'input', 'ms': 0, 'event_id': 'p1'},
            {'kind': 'commit', 'ms': 1, 'text': '车站', 'assertion': '车站'},
            {'kind': 'input', 'ms': 2, 'event_id': 'p2'},
            {'kind': 'input', 'ms': 3, 'event_id': 'p3'},
            {'kind': 'commit', 'ms': 3, 'text': '车站', 'assertion': '车站'},
            {'kind': 'commit', 'ms': 4, 'text': '更正一下', 'assertion': None},
            {'kind': 'commit', 'ms': 5, 'text': '图书馆门口', 'assertion': '图书馆门口'},
            {'kind': 'input', 'ms': 6, 'event_id': 'p4'},
        ]
        result = score_trace(self.case, trace)
        self.assertEqual(result['incorrect_assertion_commits'], 1)
        self.assertEqual(result['assertion_commits_before_last_input'], 3)
        self.assertEqual(result['unscored_commits'], 1)
        self.assertEqual([x['first_correct_after_ms'] for x in result['target_intervals']], [1, 2])

    def test_silence_is_not_a_repair(self):
        trace = [{'kind': 'input', 'event_id': e['id'], 'ms': i}
                 for i, e in enumerate(self.case['events'])]
        result = score_trace(self.case, trace)
        self.assertEqual([x['status'] for x in result['target_intervals']], ['superseded', 'unanswered'])
        with self.assertRaises(ValueError):
            score_trace(self.case, trace[:-1])
        with self.assertRaises(ValueError):
            score_trace(self.case, trace[::-1])

    def test_cannot_commit_before_first_input(self):
        with self.assertRaises(ValueError):
            score_trace(self.case, [{'kind': 'commit', 'ms': 0, 'text': '车站', 'assertion': '车站'}])
