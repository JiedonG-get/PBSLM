import unittest
import torch
from models.segmented_working_read import read_segments


class SegmentedWorkingReadTests(unittest.TestCase):
    def test_permutation_invariance_and_correct_record(self):
        query=torch.tensor([[1.,0.]])
        keys=torch.tensor([[[0.,1.],[1.,0.],[-1.,0.]]])
        values=torch.tensor([[[0.,1.],[.8,.2],[1.,0.]]])
        result=read_segments(query,keys,values)
        permutation=torch.tensor([2,0,1])
        moved=read_segments(query,keys[:,permutation],values[:,permutation])
        torch.testing.assert_close(result['value'],moved['value'])
        self.assertEqual(int(result['top_record']),1)
        self.assertEqual(int(permutation[moved['top_record']]),1)
        hard=read_segments(query,keys,values,hard=True)
        torch.testing.assert_close(hard['value'],values[:,1])

    def test_hard_tie_is_order_dependent_but_soft_value_is_not(self):
        q=torch.tensor([[1.,0.]])
        keys=torch.tensor([[[1.,0.],[1.,0.]]]);values=torch.eye(2)[None]
        a=read_segments(q,keys,values,hard=True)['value']
        b=read_segments(q,keys.flip(1),values.flip(1),hard=True)['value']
        self.assertFalse(torch.equal(a,b))
        torch.testing.assert_close(read_segments(q,keys,values)['value'],read_segments(q,keys.flip(1),values.flip(1))['value'])

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):read_segments(torch.ones(1,2),torch.empty(1,0,2),torch.empty(1,0,3))
        with self.assertRaises(ValueError):read_segments(torch.ones(1,2),torch.full((1,1,2),float('nan')),torch.ones(1,1,3))
        with self.assertRaises(ValueError):read_segments(torch.ones(1,2),torch.ones(1,1,2),torch.ones(1,1,3),temperature=0)


if __name__=='__main__':unittest.main()
