import unittest
import torch
from models.low_rank_ridge import LowRankRidge
from models.cumulative_ridge import CumulativeRidge


class LowRankRidgeTests(unittest.TestCase):
    def test_full_rank_matches_joint_ridge_across_stages(self):
        gen=torch.Generator().manual_seed(123)
        x=torch.randn(30,5,generator=gen,dtype=torch.float64);y=torch.randn(30,3,generator=gen,dtype=torch.float64)
        sketch=LowRankRidge(5,3,6);joint=CumulativeRidge(5,3)
        for start in (0,10,20):
            sketch.add_stage(x[start:start+10],y[start:start+10]);joint.add(x[start:start+10],y[start:start+10]);joint.solve()
            torch.testing.assert_close(sketch.predict(x),joint.predict(x),atol=1e-8,rtol=1e-8)

    def test_rank_bound_and_failed_update_atomic(self):
        sketch=LowRankRidge(5,3,2)
        sketch.add_stage(torch.eye(5),torch.ones(5,3))
        self.assertEqual(sketch.basis.shape[1],2)
        self.assertEqual(sketch.storage()['tensor_payload_bytes'],(6*2+2+2*3)*8)
        before=sketch.basis.clone();count=sketch.samples
        with self.assertRaises(ValueError):sketch.add_stage(torch.ones(1,5),torch.full((1,3),float('nan')))
        torch.testing.assert_close(before,sketch.basis);self.assertEqual(sketch.samples,count)


if __name__=='__main__':unittest.main()
