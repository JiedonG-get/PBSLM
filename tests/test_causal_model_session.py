"""Real PBSLMv2 + runtime + numerical M, no optimizer or training."""
import copy
import json
import unittest
from unittest.mock import patch

import torch
from models.model_v2 import PBSLMv2
from runtime.model_session import CausalModelSession, state_hash
from runtime.session import StaleProposalError, TransitionError


def make_model():
    torch.manual_seed(27)
    return PBSLMv2(vocab=48, k=8, depth=1, d=16, heads=4, m_max=4,
                    e_n=2, act_x=.5, field_scale=0., trainable_target=False,
                    use_sample_masks=True).eval()


class CausalModelSessionTests(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.session = CausalModelSession(self.model)

    def prepare(self):
        observation = self.session.observe([4, 4, 4, 4], event_time=1)
        prediction = self.session.advance()
        return observation, prediction

    def test_commit_does_not_write_memory(self):
        _, result = self.prepare()
        memory = state_hash(self.model, memory=True)
        proposal = self.session.propose(result['tokens_a'])
        text = json.dumps(result['tokens_a'])
        self.session.commit(proposal, expected_offset=0, count=len(text))
        self.assertEqual(memory, state_hash(self.model, memory=True))
        self.assertEqual(self.session.protocol.confirmed_writes, [])

    def test_feedback_confirm_is_causal_and_idempotent(self):
        obs, _ = self.prepare()
        parameters = state_hash(self.model)
        with torch.no_grad():
            c_before = self.model(torch.tensor([[4,4,4,4]]), 1, sample_noise=False)['z_c']
        original = state_hash(self.model, memory=True)
        with self.assertRaises(ValueError):
            self.session.stage_feedback(obs, [24], event_time=0)
        candidate = self.session.stage_feedback(obs, [24], event_time=2)
        self.assertEqual(original, state_hash(self.model, memory=True))
        first = self.session.confirm_memory(candidate)
        changed = state_hash(self.model, memory=True)
        self.assertNotEqual(original, changed)
        second = self.session.confirm_memory(candidate)
        self.assertEqual(first['version'], second['version'])
        self.assertTrue(second['replayed'])
        self.assertEqual(changed, state_hash(self.model, memory=True))
        self.assertEqual(parameters, state_hash(self.model))
        with torch.no_grad():
            c_after = self.model(torch.tensor([[4,4,4,4]]), 1, sample_noise=False)['z_c']
        self.assertTrue(torch.equal(c_before, c_after))

    def test_failure_restores_memory_and_confirmation(self):
        obs, _ = self.prepare()
        candidate = self.session.stage_feedback(obs, [24], event_time=2)
        before = state_hash(self.model, memory=True)
        snap = self.session.protocol.snapshot()

        def bad_write(*args, **kwargs):
            with torch.no_grad():
                self.model.mem[0].fill_(float('nan'))
            raise RuntimeError('injected failure')

        with patch.object(self.model, 'write_memory', side_effect=bad_write):
            with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                self.session.confirm_memory(candidate)
        self.assertEqual(before, state_hash(self.model, memory=True))
        self.assertEqual(snap, self.session.protocol.snapshot())
        self.assertEqual(self.session.applied, [])
        self.session.confirm_memory(candidate)

    def test_snapshot_restores_pending_and_numerical_memory(self):
        obs, _ = self.prepare()
        candidate = self.session.stage_feedback(obs, [25], event_time=2)
        pending = json.loads(json.dumps(self.session.snapshot()))
        other_model = make_model()
        restored = CausalModelSession.restore(other_model, pending)
        self.session.confirm_memory(candidate)
        restored.confirm_memory(candidate)
        self.assertEqual(state_hash(self.model, memory=True), state_hash(other_model, memory=True))
        confirmed = self.session.snapshot()
        with torch.no_grad():
            other_model.mem[0].zero_()
        again = CausalModelSession.restore(other_model, confirmed)
        self.assertTrue(again.confirm_memory(candidate)['replayed'])
        self.assertEqual(state_hash(self.model, memory=True), state_hash(other_model, memory=True))
        self.assertEqual(again.snapshot(), confirmed)

    def test_interrupt_stale_output_and_resume_prediction(self):
        _, prediction = self.prepare()
        proposal = self.session.propose(prediction['tokens_a'])
        self.session.interrupt()
        self.session.observe([5, 5, 5, 5], event_time=2)
        self.session.resume()
        next_prediction = self.session.advance()
        self.assertNotEqual(prediction['observation_id'], next_prediction['observation_id'])
        with self.assertRaises(StaleProposalError):
            self.session.commit(proposal, expected_offset=0, count=1)

    def test_reject_and_external_mutation_guard(self):
        obs, _ = self.prepare()
        candidate = self.session.stage_feedback(obs, [24], event_time=2)
        self.session.reject_memory(candidate)
        with self.assertRaises(TransitionError):
            self.session.confirm_memory(candidate)
        with torch.no_grad():
            self.model.mem[0].add_(1)
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.session.observe([4], event_time=3)

    def test_restore_wrong_model_or_bad_memory_rejected_atomically(self):
        self.prepare()
        snap = self.session.snapshot()
        before = state_hash(self.model, memory=True)
        bad = copy.deepcopy(snap)
        bad['memory'][0] = [[0]]
        with self.assertRaises(ValueError):
            CausalModelSession.restore(self.model, bad)
        self.assertEqual(before, state_hash(self.model, memory=True))
        with torch.no_grad():
            self.model.out_a.weight.add_(.01)
        with self.assertRaisesRegex(ValueError, 'model/schema'):
            CausalModelSession.restore(self.model, snap)


    def test_same_weights_different_inference_config_rejected(self):
        self.prepare()
        snap = self.session.snapshot()
        same_weights = make_model()
        same_weights.use_flow = False
        before = state_hash(same_weights, memory=True)
        with self.assertRaisesRegex(ValueError, 'model/schema'):
            CausalModelSession.restore(same_weights, snap)
        self.assertEqual(before, state_hash(same_weights, memory=True))
        same_weights = make_model()
        same_weights.dec_a[0].res_scale = 9.0
        with self.assertRaisesRegex(ValueError, 'model/schema'):
            CausalModelSession.restore(same_weights, snap)
        self.model.use_sample_masks = False
        with self.assertRaisesRegex(ValueError, 'version/mode'):
            self.session.observe([4], event_time=2)

    def test_commit_refuses_stale_model_or_external_memory(self):
        _, prediction = self.prepare()
        proposal = self.session.propose(prediction['tokens_a'])
        self.model.use_mem = False
        with self.assertRaisesRegex(ValueError, 'version/mode'):
            self.session.commit(proposal, expected_offset=0, count=1)
        self.model.use_mem = True
        with torch.no_grad():
            self.model.mem[0].add_(1)
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.session.commit(proposal, expected_offset=0, count=1)
        self.assertEqual(self.session.protocol.output, '')

    def test_confirmation_receipt_conflict_and_corrupt_capture(self):
        obs, _ = self.prepare()
        candidate = self.session.stage_feedback(obs, [24], event_time=2)
        snap = self.session.snapshot()
        corrupted = copy.deepcopy(snap)
        corrupted['pending'][candidate]['features'] = None
        before = state_hash(self.model, memory=True)
        with self.assertRaises(ValueError):
            CausalModelSession.restore(self.model, corrupted)
        self.assertEqual(before, state_hash(self.model, memory=True))
        missing = copy.deepcopy(snap)
        del missing['pending']
        with self.assertRaises(ValueError):
            CausalModelSession.restore(self.model, missing)
        receipt = self.session.confirm_memory(candidate, lr=.03)
        self.assertNotEqual(receipt['memory_before'], receipt['memory_after'])
        self.assertEqual(receipt['observation_id'], obs)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            self.session.confirm_memory(candidate, lr=.05)
        self.assertTrue(self.session.confirm_memory(candidate, lr=.03)['replayed'])


if __name__ == '__main__':
    unittest.main()
