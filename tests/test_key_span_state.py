import unittest
import torch
from models.key_span_state import KeySpanState


class KeySpanStateTests(unittest.TestCase):
    def test_observed_span_and_unseen_in_span_not_distinguished(self):
        state=KeySpanState(4);eye=torch.eye(4)
        state.observe(eye[0]);state.observe(eye[1]);state.observe(eye[0])
        self.assertEqual(state.rank,2)
        scores=state.novelty(torch.stack((eye[0],eye[2],eye[0]+eye[1])))
        torch.testing.assert_close(scores,torch.tensor([0.,1.,0.],dtype=torch.float64))
        for k in eye:state.observe(k)
        self.assertEqual(state.rank,4)
        self.assertLess(float(state.novelty(torch.randn(5,4)).max()),1e-10)

    def test_matches_explicit_complement_projector(self):
        keys=torch.randn(6,16,generator=torch.Generator().manual_seed(5),dtype=torch.float64)
        state=KeySpanState()
        for k in keys:state.observe(k)
        expected=torch.eye(16,dtype=torch.float64)-keys.T@torch.linalg.solve(keys@keys.T,keys)
        torch.testing.assert_close(state.projector,expected,atol=1e-9,rtol=1e-9)
        with self.assertRaises(ValueError):state.observe(torch.zeros(16))


if __name__=='__main__':unittest.main()
