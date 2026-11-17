import unittest
import torch
from models.content_fast_state import ContentFastState


class ContentFastStateTests(unittest.TestCase):
    def test_delta_exact_unit_gate_overwrites_orthogonal_key_without_touching_other(self):
        model=ContentFastState().eval()
        with torch.no_grad():model.write_logit.fill_(100.)
        state=model.initial(1);a=torch.eye(16)[0:1];b=torch.eye(16)[1:2]
        state=model.observe(state,a,torch.tensor([2]));state=model.observe(state,b,torch.tensor([4]))
        saved=state[:,1].clone();state=model.observe(state,a,torch.tensor([3]))
        torch.testing.assert_close(state[:,0],torch.eye(8)[3:4])
        torch.testing.assert_close(state[:,1],saved)
        self.assertEqual(state.numel(),128)

    def test_first_write_equal_and_same_key_residual_contracts(self):
        torch.manual_seed(41);delta=ContentFastState(rule='delta');additive=ContentFastState(rule='additive')
        additive.load_state_dict(delta.state_dict())
        key=torch.nn.functional.normalize(torch.randn(1,16),dim=-1);value=torch.tensor([2])
        initial=delta.initial(1)
        torch.testing.assert_close(delta.observe(initial,key,value),additive.observe(initial,key,value))
        state=torch.randn(1,16,8)
        before=torch.einsum('bd,bdv->bv',key,state)
        updated=delta.observe(state,key,value)
        after=torch.einsum('bd,bdv->bv',key,updated)
        target=delta.value_embedding(value);beta=delta.write_logit.sigmoid()
        torch.testing.assert_close(after,(1-beta)*before+beta*target)

    def test_query_read_only_and_gradients(self):
        model=ContentFastState();key=torch.randn(2,16);values=torch.tensor([1,4])
        state=model.observe(model.initial(2),key,values);saved=state.detach().clone()
        output=model.query(state,key)
        self.assertTrue(torch.equal(state.detach(),saved))
        torch.nn.functional.cross_entropy(output,values).backward()
        self.assertGreater(float(model.value_embedding.weight.grad.abs().sum()),0.)
        self.assertIsNotNone(model.write_logit.grad)
        self.assertEqual(set(dict(model.named_children())),{'value_embedding','read'})


if __name__=='__main__':unittest.main()
