import unittest
import torch
from models.continuous_key_state import ContinuousKeyState
from evaluation.continuous_key_state_control import train_episode,evaluate_episode


class ContinuousKeyTaskTests(unittest.TestCase):
    def test_episode_has_unknown_queries_and_explicit_corrections(self):
        model=ContinuousKeyState(width=24).eval();generator=torch.Generator().manual_seed(6)
        with torch.no_grad():
            result=evaluate_episode(model,32,24,generator,'cpu',.01,.8)
        self.assertEqual(result['memory_keys'],8)
        self.assertEqual(result['queries'],32*(24-8))
        self.assertGreater(result['correction_writes'],0)
        self.assertGreater(result['metrics']['state']['unknown_n'],0)
        self.assertGreater(result['metrics']['state']['known_n'],0)

    def test_training_episode_produces_finite_loss_and_gradients(self):
        torch.manual_seed(9);model=ContinuousKeyState(width=24)
        generator=torch.Generator().manual_seed(19)
        loss=train_episode(model,16,24,generator,'cpu')
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.update.weight_ih.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
