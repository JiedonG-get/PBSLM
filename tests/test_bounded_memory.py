"""Bounded external ledger tests; no training or optimizer operations."""
import copy
import json
import unittest
from unittest.mock import patch

import torch
from models.model_v2 import PBSLMv2
from runtime.bounded_memory import BoundedFeedbackMemory
from runtime.model_session import state_hash


def model():
    torch.manual_seed(31)
    return PBSLMv2(vocab=40, k=8, depth=1, d=16, heads=4, e_n=2,
                    m_max=2, act_x=.5, trainable_target=False,
                    use_sample_masks=True, field_scale=0).eval()


def features(m, key):
    with torch.no_grad():
        return m.e_forward(m.token_emb(torch.tensor([[key] * 4])))[0]


class BoundedMemoryTests(unittest.TestCase):
    def setUp(self):
        self.m = model()
        self.c = BoundedFeedbackMemory(self.m, 2)
        self.f = features(self.m, 4)

    def test_overwrite_removes_old_contribution_and_c_is_stable(self):
        x = torch.tensor([[4] * 4])
        with torch.no_grad():
            c_before = self.m(x, 1, sample_noise=False)['z_c'].clone()
        fixed = state_hash(self.m)
        self.c.put_confirmed('e1', 'fact/4', self.f, [24])
        self.c.put_confirmed('e2', 'fact/4', self.f, [25])
        expected = model()
        with torch.no_grad():
            expected.write_memory(self.f, expected.target_embed(torch.tensor([[25]])), lr=.05)
        for actual, wanted in zip(self.m.mem, expected.mem):
            torch.testing.assert_close(actual, wanted)
        self.assertEqual(self.c.slots, ('fact/4',))
        self.assertEqual(state_hash(self.m), fixed)
        with torch.no_grad():
            self.assertTrue(torch.equal(c_before, self.m(x, 1, sample_noise=False)['z_c']))

    def test_slot_or_order_only_corruption_rejected(self):
        self.c.put_confirmed('1', 'a', self.f, [24])
        self.c.put_confirmed('2', 'b', features(self.m,5), [25])
        snap=self.c.snapshot()
        before=state_hash(self.m,memory=True)
        for mutation in ('slot','order'):
            bad=copy.deepcopy(snap)
            if mutation=='slot':
                bad['ledger'][0][0]='renamed-slot'
            else:
                bad['ledger'].reverse()
            with self.assertRaisesRegex(ValueError,'ledger.*history'):
                BoundedFeedbackMemory.restore(self.m,bad)
            self.assertEqual(before,state_hash(self.m,memory=True))

    def test_lru_update_recency_and_fifo_order(self):
        for policy, expected in [('lru', ('a', 'c')), ('fifo', ('b', 'c'))]:
            with self.subTest(policy=policy):
                m = model()
                controller = BoundedFeedbackMemory(m, 2, policy=policy)
                for event, slot in [('1', 'a'), ('2', 'b'), ('3', 'a'), ('4', 'c')]:
                    receipt = controller.put_confirmed(event, slot, self.f, [24])
                self.assertEqual(controller.slots, expected)
                self.assertEqual(receipt['evicted'], 'b' if policy == 'lru' else 'a')

    def test_remove_rebuild_and_noop(self):
        self.c.put_confirmed('1', 'a', self.f, [24])
        self.c.put_confirmed('2', 'b', features(self.m, 5), [25])
        self.c.remove('3', 'a')
        expected = model()
        with torch.no_grad():
            expected.write_memory(features(expected, 5), expected.target_embed(torch.tensor([[25]])), lr=.05)
        for a, b in zip(self.m.mem, expected.mem):
            torch.testing.assert_close(a, b)
        self.c.remove('4', 'b')
        self.assertTrue(all(m.count_nonzero() == 0 for m in self.m.mem))
        self.c.remove('5', 'not-present')
        self.assertEqual(self.c.slots, ())

    def test_idempotence_conflicts_capture_and_receipt_copy(self):
        first = self.c.put_confirmed('1', 'a', self.f, [24])
        before = state_hash(self.m, memory=True)
        self.assertEqual(first, self.c.put_confirmed('1', 'a', self.f.clone(), [24]))
        self.assertEqual(before, state_hash(self.m, memory=True))
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            self.c.put_confirmed('1', 'a', self.f, [25])
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            self.c.remove('1', 'a')
        first['slot'] = 'changed'
        history = self.c.receipts
        history[0]['retained_slots'].clear()
        self.assertEqual(self.c.receipts[0]['retained_slots'], ['a'])
        self.f.zero_()  # confirmation captured a detached copy
        self.c.put_confirmed('2', 'b', features(self.m, 5), [26])
        self.assertEqual(self.c.receipts[0]['slot'], 'a')

    def test_put_and_remove_failure_roll_back_everything(self):
        self.c.put_confirmed('1', 'a', self.f, [24])
        self.c.put_confirmed('2', 'b', self.f, [25])
        before = self.c.snapshot()

        def failure(*args, **kwargs):
            with torch.no_grad():
                self.m.mem[0].fill_(float('nan'))
            raise RuntimeError('injected write failure')

        for action in [lambda: self.c.put_confirmed('3', 'c', self.f, [26]),
                       lambda: self.c.remove('3', 'a')]:
            with patch.object(self.m, 'write_memory', side_effect=failure):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    action()
            self.assertEqual(self.c.snapshot(), before)
        self.c.remove('3', 'a')  # failed event ID was not consumed

    def test_nonfinite_write_without_exception_rolls_back(self):
        self.c.put_confirmed('1', 'a', self.f, [24])
        before = self.c.snapshot()

        def corrupt(*args, **kwargs):
            with torch.no_grad():
                self.m.mem[0].fill_(float('inf'))

        with patch.object(self.m, 'write_memory', side_effect=corrupt):
            with self.assertRaises(FloatingPointError):
                self.c.put_confirmed('2', 'b', self.f, [25])
        self.assertEqual(self.c.snapshot(), before)

    def test_snapshot_restore_recomputes_then_installs(self):
        self.c.put_confirmed('1', 'a', self.f, [24])
        self.c.put_confirmed('2', 'b', self.f, [25])
        snapshot = json.loads(json.dumps(self.c.snapshot()))
        m = model()
        with torch.no_grad():
            m.mem[0].fill_(.7)  # restoration need not start from zero
        restored = BoundedFeedbackMemory.restore(m, snapshot)
        self.assertEqual(restored.snapshot(), snapshot)
        self.assertEqual(restored.put_confirmed('1', 'a', self.f, [24]), self.c.receipts[0])
        restored.remove('3', 'b')
        self.c.remove('3', 'b')
        self.assertEqual(restored.snapshot(), self.c.snapshot())

    def test_snapshot_mismatch_and_rebuild_failure_atomic(self):
        self.c.put_confirmed('1', 'a', self.f, [24])
        snapshot = self.c.snapshot()
        bad = copy.deepcopy(snapshot)
        bad['memory'][0][0][0] += 1
        before = state_hash(self.m, memory=True)
        with self.assertRaisesRegex(ValueError, 'reconstruct'):
            BoundedFeedbackMemory.restore(self.m, bad)
        self.assertEqual(before, state_hash(self.m, memory=True))
        bad = copy.deepcopy(snapshot)
        bad['ledger'][0][1]['features'] = [[1]]
        with self.assertRaises(ValueError):
            BoundedFeedbackMemory.restore(self.m, bad)
        self.assertEqual(before, state_hash(self.m, memory=True))
        bad = copy.deepcopy(snapshot)
        bad['memory'][0] = [[0]]
        with self.assertRaisesRegex(ValueError, 'numerical'):
            BoundedFeedbackMemory.restore(self.m, bad)
        self.assertEqual(before, state_hash(self.m, memory=True))
        bad = copy.deepcopy(snapshot)
        bad['events']['1']['request']['payload']['label_ids'] = [25]
        with self.assertRaisesRegex(ValueError, 'payload hash'):
            BoundedFeedbackMemory.restore(self.m, bad)
        self.assertEqual(before, state_hash(self.m, memory=True))
        with patch.object(self.m, 'write_memory', side_effect=RuntimeError('restore write')):
            with self.assertRaisesRegex(RuntimeError, 'restore write'):
                BoundedFeedbackMemory.restore(self.m, snapshot)
        self.assertEqual(before, state_hash(self.m, memory=True))
        other = model()
        other.use_flow = False
        with self.assertRaisesRegex(ValueError, 'model/schema'):
            BoundedFeedbackMemory.restore(other, snapshot)

    def test_guard_and_validation(self):
        with self.assertRaises(ValueError):
            self.c.put_confirmed('1', '', self.f, [24])
        with self.assertRaises(ValueError):
            self.c.put_confirmed('1', 'a', torch.full_like(self.f, float('nan')), [24])
        with self.assertRaises(ValueError):
            self.c.put_confirmed('1', 'a', self.f, [-1])
        with torch.no_grad():
            self.m.mem[0].add_(1)
        with self.assertRaisesRegex(ValueError, 'external'):
            self.c.remove('1', 'a')
        with self.assertRaisesRegex(ValueError, 'zero memory'):
            BoundedFeedbackMemory(self.m)
        self.m = model()
        controller = BoundedFeedbackMemory(self.m)
        self.m.direct_only = True
        with self.assertRaisesRegex(ValueError, 'configuration'):
            controller.put_confirmed('1', 'a', self.f, [24])


if __name__ == '__main__':
    unittest.main()
