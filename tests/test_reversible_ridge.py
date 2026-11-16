import unittest
from unittest.mock import patch
import torch
from models.reversible_ridge import ReversibleRidge
from models.cumulative_ridge import CumulativeRidge


class ReversibleRidgeTests(unittest.TestCase):
    def test_replace_remove_equal_latest_only_rebuild(self):
        model=ReversibleRidge(3,2,capacity=2)
        x=torch.eye(3,dtype=torch.float64)
        model.put('a',x[:1],torch.tensor([[1.,0.]]))
        model.put('b',x[1:2],torch.tensor([[0.,1.]]))
        new=torch.tensor([[.2,.8]],dtype=torch.float64)
        model.put('a',x[:1],new)
        expected=CumulativeRidge(3,2);expected.add(x[1:2],torch.tensor([[0.,1.]]));expected.add(x[:1],new);expected.solve()
        torch.testing.assert_close(model.model.weight,expected.weight)
        model.remove('b')
        expected=CumulativeRidge(3,2);expected.add(x[:1],new);expected.solve()
        torch.testing.assert_close(model.model.weight,expected.weight)
        model.remove('a');self.assertEqual(model.model.samples,0)
        self.assertEqual(int(model.model.weight.count_nonzero()),0)

    def test_failure_atomic_capture_and_update_recency_eviction(self):
        model=ReversibleRidge(2,2,capacity=2);x=torch.eye(2);y=x.clone()
        model.put('a',x[:1],y[:1]);old=model.model.weight.clone();version=model.version
        with patch.object(model,'_rebuild',side_effect=RuntimeError('solve failed')):
            with self.assertRaises(RuntimeError):model.put('b',x[1:],y[1:])
        self.assertEqual(model.version,version);self.assertEqual(model.slots,('a',))
        torch.testing.assert_close(old,model.model.weight)
        model.put('b',x[1:],y[1:]);model.put('a',x[:1],y[:1])
        result=model.put('c',x[:1],y[:1]);self.assertEqual(result['evicted'],'b')
        x.fill_(99);y.fill_(99)
        self.assertLess(float(model._slots['a'][0].max()),2.)
        self.assertEqual(model.storage()['retained_feature_target_rows'],2)
        with self.assertRaises(ValueError):model.put('too-many',torch.zeros(9,2),torch.zeros(9,2))
        with self.assertRaises(ValueError):model.put('bad',torch.zeros(1,3),torch.zeros(1,2))


if __name__=='__main__':unittest.main()
