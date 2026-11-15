"""Bounded slot correction through Runtime confirmations; no training."""
import copy
import json
import unittest
from unittest.mock import patch

import torch
from runtime.bounded_session import BoundedCausalModelSession
from runtime.model_session import state_hash
import tests.test_causal_model_session as helpers


class BoundedSessionTests(unittest.TestCase):
    def setUp(self):
        self.model = helpers.make_model()
        self.session = BoundedCausalModelSession(self.model, capacity=2)
        self.clock = 0

    def feedback(self, key, label, slot):
        self.clock += 1
        obs = self.session.observe([key]*4,event_time=self.clock)
        self.session.advance()
        self.clock += 1
        return self.session.stage_feedback(obs,[label],event_time=self.clock,slot=slot)

    def test_confirm_overwrite_and_eviction(self):
        c1=self.feedback(4,24,'device4/color')
        self.assertEqual(self.session.controller.slots,())
        self.session.confirm_memory(c1)
        c2=self.feedback(4,25,'device4/color')
        receipt=self.session.confirm_memory(c2)
        self.assertEqual(self.session.controller.slots,('device4/color',))
        expected=helpers.make_model()
        with torch.no_grad():
            feat,_=expected.e_forward(expected.token_emb(torch.tensor([[4]*4])))
            expected.write_memory(feat,expected.target_embed(torch.tensor([[25]])),lr=.05)
        for a,b in zip(self.model.mem,expected.mem):
            torch.testing.assert_close(a,b)
        again=self.session.confirm_memory(c2)
        self.assertTrue(again['replayed'])
        self.assertEqual(receipt['memory_after'],again['memory_after'])
        self.session.confirm_memory(self.feedback(5,26,'device5/color'))
        evicted=self.session.confirm_memory(self.feedback(6,27,'device6/color'))
        self.assertEqual(evicted['evicted'],'device4/color')
        self.assertEqual(len(self.session.controller.slots),2)

    def test_failure_after_numerical_update_restores_both_histories(self):
        candidate=self.feedback(4,24,'slot')
        before=self.session.snapshot()
        with patch.object(self.session.protocol,'confirm_write',side_effect=RuntimeError('confirmation failed')):
            with self.assertRaisesRegex(RuntimeError,'confirmation failed'):
                self.session.confirm_memory(candidate)
        self.assertEqual(before,self.session.snapshot())
        self.session.confirm_memory(candidate)

    def test_snapshot_correction_continuation(self):
        c1=self.feedback(4,24,'slot')
        self.session.confirm_memory(c1)
        c2=self.feedback(4,25,'slot')
        snap=json.loads(json.dumps(self.session.snapshot()))
        model=helpers.make_model()
        restored=BoundedCausalModelSession.restore(model,snap)
        left=self.session.confirm_memory(c2)
        right=restored.confirm_memory(c2)
        self.assertEqual(left,right)
        self.assertEqual(state_hash(model,memory=True),state_hash(self.model,memory=True))
        self.assertEqual(self.session.snapshot(),restored.snapshot())

    def test_mismatched_snapshots_fail_atomically(self):
        candidate=self.feedback(4,24,'slot')
        self.session.confirm_memory(candidate)
        snap=self.session.snapshot()
        before=state_hash(self.model,memory=True)
        bad=copy.deepcopy(snap)
        bad['base']['memory'][0][0][0] += 1
        with self.assertRaises(ValueError):
            BoundedCausalModelSession.restore(self.model,bad)
        self.assertEqual(before,state_hash(self.model,memory=True))

    def test_rejected_candidate_never_enters_ledger(self):
        c1=self.feedback(4,24,'slot')
        self.session.reject_memory(c1)
        with self.assertRaises(RuntimeError):
            self.session.confirm_memory(c1)
        self.assertEqual(self.session.controller.slots,())


if __name__=='__main__': unittest.main()
