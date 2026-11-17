import unittest
import torch
from models.sparse_working_state import SparseWorkingState


class SparseWorkingStateTests(unittest.TestCase):
    def test_train_sparse_forward_and_route_gradients(self):
        torch.manual_seed(13);model=SparseWorkingState().train()
        state=torch.randn(2,8,12);ids=torch.tensor([0,1]);values=torch.tensor([2,3])
        top=model.routing(ids).argmax(-1)
        result=model.observe(state,ids,values)
        for b in range(2):
            for slot in range(8):
                if slot!=int(top[b]):self.assertTrue(torch.equal(result[b,slot],state[b,slot]))
        loss=torch.nn.functional.cross_entropy(model.query(result,ids),values)+.1*model.collision_loss()
        loss.backward()
        self.assertGreater(float(model.keys.grad.abs().sum()),0.)
        self.assertGreater(float(model.cell.weight_ih.grad.abs().sum()),0.)

    def test_surrogate_gradient_through_write_without_read_or_prior(self):
        torch.manual_seed(29);model=SparseWorkingState().train()
        state=torch.randn(2,8,12);ids=torch.tensor([0,1]);values=torch.tensor([2,3])
        result=model.observe(state,ids,values)
        probe=torch.linspace(-1,1,result.numel()).reshape_as(result)
        (result*probe).sum().backward()
        self.assertGreater(float(model.keys.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in model.read.parameters()))

    def test_eval_unaddressed_slots_exact_and_prior_finite(self):
        model=SparseWorkingState().eval();state=torch.randn(2,8,12)
        ids=torch.tensor([2,3]);values=torch.tensor([4,5])
        with torch.no_grad():
            top=model.routing(ids).argmax(-1);result=model.observe(state,ids,values)
            self.assertTrue(torch.isfinite(model.collision_loss()))
            for b in range(2):
                for slot in range(8):
                    if slot!=int(top[b]):self.assertTrue(torch.equal(state[b,slot],result[b,slot]))


if __name__=='__main__':unittest.main()
