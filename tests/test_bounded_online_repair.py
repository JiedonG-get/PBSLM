import asyncio
import time
import unittest

from evaluation.bounded_online_memory import make_memory
from evaluation.bounded_online_repair import CASES, ORACLES, run_case, score_run


class ImmediateBackend:
    def __init__(self, model, max_new_tokens, *, render_messages):
        self.render_messages = render_messages
        self.started = asyncio.Event()
        self.metrics = []

    async def generate(self, request):
        started = time.perf_counter()
        self.started.set()
        messages = self.render_messages(request)
        answer = messages[-1]['content'].rsplit('：', 1)[1]
        self.metrics.append({'revision': request.revision, 'messages': messages, 'answer': answer,
                             'start_perf': started, 'finish_perf': time.perf_counter()})
        yield answer

    async def close(self):
        pass


class IgnoreCorrections:
    def __init__(self, arm, *, capacity):
        self.memory = make_memory(arm, capacity=capacity)
        self.answer = self.memory.answer
        self.peek = self.memory.peek
        self.metrics = self.memory.metrics
        self.has_value = False

    def observe(self, text):
        if not self.has_value:
            self.has_value = self.memory.observe(text)


class BoundedOnlineRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_pending_old_answer_is_cancelled_before_delivery(self):
        case = CASES[0]
        before, after = ORACLES[case.name]
        run = await run_case(None, case, 'exact_bounded', backend_factory=ImmediateBackend)
        scores = score_run(run)
        self.assertEqual(run['prefix_at_update'], '')
        self.assertEqual(run['worker_jobs'][0]['answer'], f'{before}。')
        self.assertEqual(run['worker_jobs'][1]['answer'], f'{after}。')
        self.assertEqual([commit['unit'] for commit in run['committed']], [f'{after}。'])
        self.assertFalse(run['initial_worker_active_at_update'])
        self.assertFalse(scores['initial_committed_after_update'])
        self.assertTrue(scores['replacement_current_label_present'])
        self.assertEqual(scores['update_cancel_attempts'], 1)
        self.assertEqual(scores['commits_with_only_outdated_label'], 0)

    async def test_missed_correction_releases_a_stale_answer(self):
        run = await run_case(None, CASES[0], 'exact_bounded', backend_factory=ImmediateBackend,
                             memory_factory=IgnoreCorrections)
        scores = score_run(run)
        self.assertEqual(run['route_trace'][-1]['route'], 'background')
        self.assertTrue(scores['missed_replacement'])
        self.assertFalse(scores['replacement_started'])
        self.assertTrue(scores['initial_committed_after_update'])
        self.assertEqual(scores['update_cancel_attempts'], 0)
        self.assertEqual(scores['commits_with_only_outdated_label'], 1)

    async def test_postcommit_replacement_receives_and_preserves_spoken_prefix(self):
        case = CASES[3]
        before, after = ORACLES[case.name]
        run = await run_case(None, case, 'exact_unbounded', backend_factory=ImmediateBackend)
        scores = score_run(run)
        self.assertEqual(run['prefix_at_update'], f'{before}。')
        self.assertTrue(run['prefix_preserved'])
        self.assertEqual([commit['unit'] for commit in run['committed']], [f'{before}。', f'{after}。'])
        self.assertIn({'role': 'assistant', 'content': f'{before}。'}, run['worker_jobs'][1]['messages'])
        self.assertFalse(scores['commit_scores'][0]['after_update'])
        self.assertEqual(scores['commits_with_only_outdated_label'], 0)


if __name__ == '__main__':
    unittest.main()
