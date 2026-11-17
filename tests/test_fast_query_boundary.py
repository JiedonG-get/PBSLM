import unittest
import torch
from evaluation.fast_query_boundary import nearby_key,confusion


class QueryBoundaryTests(unittest.TestCase):
    def test_nearby_norm_and_angle(self):
        gen=torch.Generator().manual_seed(4)
        k=torch.nn.functional.normalize(torch.randn(2,8,16,generator=gen),dim=-1)
        raw=torch.randn(2,8,16,generator=gen)
        for cosine in (.95,.99):
            q=nearby_key(k,raw,cosine)
            torch.testing.assert_close(q.norm(dim=-1),torch.ones(2,8),atol=1e-6,rtol=1e-6)
            torch.testing.assert_close((q*k).sum(-1),torch.full((2,8),cosine),atol=1e-6,rtol=1e-6)
        with self.assertRaises(ValueError):nearby_key(k,raw,1.)

    def test_confusion_keeps_reject_and_wrong_value_separate(self):
        c=confusion(torch.tensor([[1,8,3,2,8]]),torch.tensor([[1,2,4,8,8]]))
        self.assertEqual(c,{'n':5,'known_n':3,'known_correct':1,'known_false_unknown':1,
                           'known_wrong_value':1,'unknown_n':2,'unknown_false_accept':1,'unknown_correct_reject':1})


if __name__=='__main__':unittest.main()
