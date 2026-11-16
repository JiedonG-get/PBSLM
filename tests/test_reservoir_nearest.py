import unittest
import torch
from models.reservoir_nearest import ReservoirNearest


class ReservoirNearestTests(unittest.TestCase):
    def test_payload_bound_identity_and_reproducible_sampling(self):
        x=torch.eye(8,dtype=torch.float64);y=torch.arange(8,dtype=torch.int64)
        full=ReservoirNearest(8,8*(8*8+8));full.add(x,y)
        self.assertTrue(torch.equal(full.predict(x),y))
        a=ReservoirNearest(8,3*(8*8+8),seed=7);b=ReservoirNearest(8,3*(8*8+8),seed=7)
        a.add(x,y);b.add(x,y)
        self.assertEqual(a.storage()['tensor_payload_bytes'],216)
        self.assertEqual(a.seen,8);self.assertEqual(a.used,3)
        self.assertTrue(torch.equal(a.x,b.x));self.assertTrue(torch.equal(a.y,b.y))

    def test_bad_input_does_not_mutate(self):
        model=ReservoirNearest(2,100)
        with self.assertRaises(ValueError):model.add(torch.ones(1,2),torch.tensor([-1]))
        self.assertEqual(model.seen,0)
        with self.assertRaises(ValueError):model.predict(torch.ones(1,2))


if __name__=='__main__':unittest.main()
