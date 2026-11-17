import unittest
import torch
from models.slotted_working_state import SlottedWorkingState


class SlottedWorkingStateTests(unittest.TestCase):
    def test_explicit_write_leaves_other_slots_exactly_unchanged(self):
        torch.manual_seed(5);model=SlottedWorkingState(addressing='explicit')
        state=torch.randn(2,8,12);saved=state.clone()
        updated=model.observe(state,torch.tensor([1,5]),torch.tensor([2,3]))
        for row,slot in enumerate((1,5)):
            self.assertFalse(torch.equal(updated[row,slot],saved[row,slot]))
            for other in range(8):
                if other!=slot:self.assertTrue(torch.equal(updated[row,other],saved[row,other]))
        before=updated.detach().clone();out=model.query(updated,torch.tensor([1,5]))
        self.assertEqual(tuple(out.shape),(2,8));self.assertTrue(torch.equal(updated.detach(),before))

    def test_learned_routing_has_gradients_and_fixed_payload(self):
        model=SlottedWorkingState();state=model.initial(3)
        updated=model.observe(state,torch.tensor([0,1,2]),torch.tensor([3,4,5]))
        self.assertEqual(updated.numel(),3*96)
        route=model.routing(torch.tensor([0,1,2]))
        torch.testing.assert_close(route.sum(-1),torch.ones(3))
        torch.nn.functional.cross_entropy(model.query(updated,torch.tensor([0,1,2])),torch.tensor([3,4,5])).backward()
        self.assertGreater(float(model.keys.grad.abs().sum()),0.)
        self.assertGreater(float(model.cell.weight_ih.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
