import unittest
import torch
from models.ridge_iterative import solve_cg


class IterativeRidgeTests(unittest.TestCase):
    def test_same_regularized_objective_as_direct_solve(self):
        generator=torch.Generator().manual_seed(84)
        x=torch.randn(40,12,generator=generator,dtype=torch.float64)
        y=torch.randn(40,5,generator=generator,dtype=torch.float64)
        gram=x.T@x;cross=x.T@y
        expected=torch.linalg.solve(gram+.01*torch.eye(12,dtype=x.dtype),cross)
        actual,metric=solve_cg(gram,cross)
        self.assertTrue(metric['converged'])
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-7)
        zero,metric=solve_cg(gram,torch.zeros_like(cross))
        self.assertEqual(metric['iterations'],0);self.assertEqual(int(zero.count_nonzero()),0)

    def test_invalid_contracts(self):
        with self.assertRaises(ValueError):solve_cg(torch.zeros(2,3),torch.zeros(2,1))
        with self.assertRaises(ValueError):solve_cg(torch.tensor([[1.,2.],[0.,1.]]),torch.ones(2,1))
        with self.assertRaises(ValueError):solve_cg(torch.eye(2),torch.ones(2,1),regularization=0)


if __name__=='__main__':unittest.main()
