import unittest
import torch
from models.dual_key_reference import DualKeyReference
from evaluation.fast_state_geometry import geometry_keys


class DualKeyReferenceTests(unittest.TestCase):
    def test_correlated_keys_overwrite_preserves_other_reads(self):
        raw=torch.randn(2,9,16,generator=torch.Generator().manual_seed(21))
        keys=geometry_keys(raw,'correlated').double();values=torch.randn(2,8,4,dtype=torch.float64)
        ref=DualKeyReference(keys,values)
        torch.testing.assert_close(ref.read(keys),values,atol=1e-9,rtol=1e-9)
        new=torch.randn(2,4,dtype=torch.float64);expected=values.clone();expected[:,0]=new
        ref.overwrite(0,new)
        torch.testing.assert_close(ref.read(keys),expected,atol=1e-9,rtol=1e-9)
        direct=DualKeyReference(keys,expected)
        torch.testing.assert_close(ref.matrix,direct.matrix,atol=1e-9,rtol=1e-9)
        new.fill_(999);self.assertFalse(torch.equal(ref.values[:,0],new))

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):DualKeyReference(torch.ones(1,2,4),torch.ones(1,2,3))


if __name__=='__main__':unittest.main()
