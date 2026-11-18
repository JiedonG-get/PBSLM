import time
import unittest

from evaluation.fixtures.postcommit_repair_contract_v1 import FORMAL
from evaluation.postcommit_repair_contract import ARMS, DeliveryBackend, Snapshot, run_case, score, template
from runtime.continuous import GenerationRequest


class ScriptedBackend:
    def __init__(self, responses, max_new_tokens, *, render_messages):
        self.responses = iter(responses)
        self.render_messages = render_messages
        self.metrics = []

    async def generate(self, request):
        messages = self.render_messages(request)
        candidate = next(self.responses)
        self.metrics.append({'revision': request.revision, 'messages': messages, 'answer': candidate})
        if candidate:
            yield candidate

    async def close(self):
        pass


class PostcommitContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_guard_falls_back_and_duplicate_deliveries_do_not_commit(self):
        case = next(c for c in FORMAL if c.identity == 'idempotent_delivery')
        run = await run_case(('雨燕厅。', template('江柳门', '雨燕厅')), case.events, 'guarded_commit',
                             backend_factory=ScriptedBackend)
        self.assertTrue(all(item['strict_pass'] for item in score(case, run)))
        self.assertEqual([item['unit'] for item in run['committed']], ['雨燕厅。', template('雨燕厅', '江柳门')])
        self.assertEqual(run['delivery_decisions'][1]['reason'], 'template_fallback')
        self.assertEqual([item['route'] for item in run['trace']][-3:],
                         ['duplicate_delivery', 'background', 'background'])

    async def test_unknown_resolution_remains_separate_from_withdrawal(self):
        case = next(c for c in FORMAL if c.identity == 'new_evidence')
        run = await run_case(('不确定。',), case.events, 'deterministic_template', backend_factory=ScriptedBackend)
        self.assertTrue(all(item['strict_pass'] for item in score(case, run)))
        self.assertEqual(len(run['worker_jobs']), 1)
        self.assertEqual([item['unit'] for item in run['committed']],
                         ['不确定。', template(None, '江柳门'), template('江柳门', '雨燕厅')])

    async def test_guard_does_not_withdraw_an_unrecognized_prior_answer(self):
        case = FORMAL[0]
        run = await run_case(('我猜是石桥厅。',), case.events, 'guarded_commit', backend_factory=ScriptedBackend)
        self.assertEqual(run['delivery_decisions'][1]['reason'], 'no_eligible_claim')
        self.assertEqual([item['unit'] for item in run['committed']], ['我猜是石桥厅。'])
        self.assertFalse(score(case, run)[-1]['strict_pass'])

    async def test_live_guard_refuses_an_outdated_snapshot(self):
        snapshot = Snapshot('请问天文角辛四号样件的当前存放位置？', '江柳门', False, 'location', '石桥厅', 1)
        candidate = template('石桥厅', '江柳门')
        backend = ScriptedBackend((candidate,), 48, render_messages=lambda request: [])
        delivery = DeliveryBackend(backend, {3: snapshot}, 'guarded_commit', lambda state: False)
        request = GenerationRequest(3, (), '石桥厅。', time.perf_counter())
        self.assertEqual([unit async for unit in delivery.generate(request)], [])
        self.assertEqual(delivery.decisions[0]['reason'], 'snapshot_no_longer_current')

    async def test_verified_structured_candidate_can_commit(self):
        case = FORMAL[0]
        valid = template('石桥厅', '江柳门')
        run = await run_case(('石桥厅。', valid), case.events, 'guarded_commit', backend_factory=ScriptedBackend)
        self.assertEqual(run['delivery_decisions'][1]['reason'], 'verified_model')
        self.assertEqual(run['committed'][1]['unit'], valid)
        self.assertEqual(len(ARMS), 4)


if __name__ == '__main__':
    unittest.main()
