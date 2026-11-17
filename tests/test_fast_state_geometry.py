import unittest
import torch
from evaluation.fast_state_geometry import geometry_keys,confusion


class FastGeometryTests(unittest.TestCase):
    def test_key_geometry_and_confusion_partition(self):
        raw=torch.randn(4,9,16,generator=torch.Generator().manual_seed(5))
        identity=torch.eye(8)[None].expand(4,-1,-1)
        orthogonal=geometry_keys(raw,'orthogonal');correlated=geometry_keys(raw,'correlated')
        torch.testing.assert_close(orthogonal@orthogonal.transpose(1,2),identity,atol=1e-6,rtol=1e-6)
        expected=.9*torch.ones(4,8,8)+.1*identity
        torch.testing.assert_close(correlated@correlated.transpose(1,2),expected,atol=1e-6,rtol=1e-6)
        metric=confusion(torch.tensor([1,8,3]),torch.tensor([1,2,4]))
        self.assertEqual(metric,{'n':3,'correct':1,'false_unknown':1,'wrong_value':1})


if __name__=='__main__':unittest.main()
