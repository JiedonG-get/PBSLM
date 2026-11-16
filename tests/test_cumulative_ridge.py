import unittest
import torch
from models.cumulative_ridge import CumulativeRidge


class CumulativeRidgeTests(unittest.TestCase):
    def test_incremental_stats_equal_joint_ridge_and_permuted_order(self):
        gen=torch.Generator().manual_seed(45)
        x=torch.randn(12,5,generator=gen,dtype=torch.float64)
        y=torch.randn(12,3,generator=gen,dtype=torch.float64)
        model=CumulativeRidge(5,3)
        for a,b in zip(x,y):model.add(a[None],b[None])
        weight=model.solve()
        augmented=torch.cat((x,torch.ones(12,1,dtype=x.dtype)),dim=1)
        expected=torch.linalg.solve(augmented.T@augmented+.01*torch.eye(6,dtype=x.dtype),augmented.T@y)
        torch.testing.assert_close(weight,expected,atol=1e-10,rtol=1e-10)
        reverse=CumulativeRidge(5,3);reverse.add(x.flip(0),y.flip(0));reverse.solve()
        torch.testing.assert_close(model.predict(x),reverse.predict(x),atol=1e-10,rtol=1e-10)
        self.assertEqual(model.samples,12)

    def test_invalid_update_does_not_change_stats(self):
        model=CumulativeRidge(2,3)
        for x,y in [(torch.zeros(1,2),torch.zeros(1,4)),(torch.full((1,2),float('nan')),torch.zeros(1,3))]:
            with self.assertRaises(ValueError):model.add(x,y)
        self.assertEqual(model.samples,0);self.assertEqual(int(model.gram.count_nonzero()),0)
        with self.assertRaises(ValueError):CumulativeRidge(2,3,regularization=float('nan'))


if __name__=='__main__':unittest.main()
