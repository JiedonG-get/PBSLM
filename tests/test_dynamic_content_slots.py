import unittest
import torch
from models.content_fast_state import ContentFastState
from models.dynamic_content_slots import DynamicContentSlots


class DynamicContentSlotsTests(unittest.TestCase):
    def model(self,capacity=2):return DynamicContentSlots(ContentFastState().eval(),capacity)

    def test_dynamic_allocation_overwrite_and_recency_eviction(self):
        model=self.model();keys=torch.eye(16);state=model.initial(1)
        state=model.observe(state,keys[4:5],torch.tensor([1]))
        state=model.observe(state,keys[2:3],torch.tensor([2]))
        before=state[:,1].clone()
        state=model.observe(state,keys[4:5],torch.tensor([3]))
        self.assertTrue(torch.equal(state[:,1,:25],before[:,:25]))
        torch.testing.assert_close(state[:,0,16:24],model.reader.value_embedding(torch.tensor([3])))
        state=model.observe(state,keys[7:8],torch.tensor([4]))
        retained=state[0,:,:16].argmax(-1).tolist()
        self.assertEqual(set(retained),{4,7})
        self.assertEqual(set(state[0,:,25].tolist()),{0.,1.})
        logits=model.query(state,keys[2:3]);self.assertEqual(int(logits.argmax(-1)),8)

    def test_similar_distinct_keys_merge_under_declared_threshold(self):
        model=self.model();state=model.initial(1);eye=torch.eye(16)
        key_a=eye[0:1];key_b=.95*key_a+(1-.95**2)**.5*eye[1:2]
        state=model.observe(state,key_a,torch.tensor([1]))
        state=model.observe(state,key_b,torch.tensor([2]))
        self.assertEqual(int((state[:,:,24]>.5).sum()),1)
        torch.testing.assert_close(state[:,0,16:24],model.reader.value_embedding(torch.tensor([2])))

    def test_noisy_versions_can_evict_unrelated_cold_record(self):
        model=self.model();state=model.initial(1);eye=torch.eye(16)
        state=model.observe(state,eye[0:1],torch.tensor([1]))
        state=model.observe(state,eye[1:2],torch.tensor([2]))
        a=.85*eye[0:1]+(1-.85**2)**.5*eye[2:3]
        b=.85*eye[0:1]-(1-.85**2)**.5*eye[2:3]
        state=model.observe(state,a,torch.tensor([3]))
        state=model.observe(state,b,torch.tensor([4]))
        self.assertEqual(int(model.query(state,eye[1:2]).argmax(-1)),8)

    def test_invalid_inputs_do_not_modify_state(self):
        model=self.model();state=model.initial(1);saved=state.clone()
        for key,value in [(torch.zeros(1,16),torch.tensor([1])),(torch.eye(16)[:1],torch.tensor([8]))]:
            with self.assertRaises(ValueError):model.observe(state,key,value)
            self.assertTrue(torch.equal(state,saved))
        with self.assertRaises(ValueError):model.query(state,torch.full((1,16),float('nan')))

    def test_query_read_only_unknown_and_repeated_clock_bound(self):
        model=self.model();state=model.initial(1);keys=torch.eye(16)
        self.assertEqual(int(model.query(state,keys[:1]).argmax(-1)),8)
        for i in range(200):state=model.observe(state,keys[i%2:i%2+1],torch.tensor([i%8]))
        self.assertLessEqual(float(state[:,:,25].max()),1.)
        saved=state.clone();model.query(state,keys[:1]);self.assertTrue(torch.equal(saved,state))
        self.assertEqual(state.numel(),52)


if __name__=='__main__':unittest.main()
