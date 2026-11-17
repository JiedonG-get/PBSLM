import unittest
import torch
from models.slotted_working_state import SlottedWorkingState
from models.state_write_intervention import observe_with_gate


class WriteGateInterventionTests(unittest.TestCase):
    def test_soft_matches_original_and_sparse_preserves_other_slots(self):
        torch.manual_seed(3);model=SlottedWorkingState().eval()
        state=torch.randn(4,8,12);entity=torch.tensor([0,1,2,3]);values=torch.tensor([4,5,6,7])
        with torch.no_grad():
            original=model.observe(state,entity,values)
            intervention=observe_with_gate(model,state,entity,values,'soft')
            self.assertTrue(torch.equal(original,intervention))
            top=model.routing(entity).argmax(-1)
            for mode in ('top1_mass','top1_unit'):
                updated=observe_with_gate(model,state,entity,values,mode)
                for batch in range(4):
                    for slot in range(8):
                        if slot!=int(top[batch]):self.assertTrue(torch.equal(state[batch,slot],updated[batch,slot]))
        with self.assertRaises(ValueError):observe_with_gate(model,state,entity,values,'unknown')


if __name__=='__main__':unittest.main()
