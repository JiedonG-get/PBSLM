import unittest
import torch
from models.recurrent_working_state import RecurrentWorkingState


class RecurrentWorkingStateTests(unittest.TestCase):
    def test_state_fixed_size_and_query_read_only(self):
        torch.manual_seed(4);model=RecurrentWorkingState().eval()
        state=model.initial(3)
        with torch.no_grad():
            for _ in range(100):state=model.observe(state,torch.tensor([0,1,2]),torch.tensor([1,2,3]))
            saved=state.clone();output=model.query(state,torch.tensor([0,1,2]))
        self.assertEqual(tuple(state.shape),(3,96));self.assertEqual(tuple(output.shape),(3,8))
        self.assertTrue(torch.equal(state,saved));self.assertTrue(torch.isfinite(state).all())
        self.assertEqual(list(model.buffers()),[])

    def test_gradients_through_update_and_read(self):
        model=RecurrentWorkingState();state=model.initial(2)
        state=model.observe(state,torch.tensor([0,1]),torch.tensor([2,3]))
        loss=torch.nn.functional.cross_entropy(model.query(state,torch.tensor([0,1])),torch.tensor([2,3]))
        loss.backward()
        self.assertGreater(float(model.update.weight_ih.grad.abs().sum()),0.)
        self.assertGreater(float(model.read[-1].weight.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
