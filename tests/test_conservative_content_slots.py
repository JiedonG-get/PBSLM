import unittest
import torch
from models.content_fast_state import ContentFastState
from models.conservative_content_slots import ConservativeContentSlots


class ConservativeSlotTests(unittest.TestCase):
    def test_full_unmatched_deferred_and_matched_anchor_not_moved(self):
        model=ConservativeContentSlots(ContentFastState().eval(),capacity=1)
        k=torch.eye(16);s=model.initial(1)
        s,r=model.observe_receipt(s,k[:1],torch.tensor([1]))
        self.assertTrue(bool(r['accepted'][0]))
        noisy=.95*k[:1]+(1-.95**2)**.5*k[1:2]
        updated,r=model.observe_receipt(s,noisy,torch.tensor([2]))
        self.assertTrue(bool(r['accepted'][0]));self.assertTrue(torch.equal(updated[:,:,:16],s[:,:,:16]))
        torch.testing.assert_close(updated[:,0,16:24],model.reader.value_embedding(torch.tensor([2])))
        rejected,r=model.observe_receipt(updated,k[2:3],torch.tensor([3]))
        self.assertTrue(bool(r['deferred'][0]));self.assertTrue(torch.equal(rejected,updated))


if __name__=='__main__':unittest.main()
