import unittest
import torch
from evaluation.text_key_transfer import projection_matrix,counts


class TextKeyTransferTests(unittest.TestCase):
    def test_fixed_projection_independent_of_data_and_orthonormal_columns(self):
        a=projection_matrix(3);b=projection_matrix(3)
        self.assertTrue(torch.equal(a,b));self.assertEqual(tuple(a.shape),(96,16))
        torch.testing.assert_close(a.T@a,torch.eye(16),atol=1e-6,rtol=1e-6)
        self.assertFalse(torch.equal(a,projection_matrix(4)))

    def test_no_unknown_queries_not_divided_by_zero_or_counted_correct(self):
        metric=counts(torch.tensor([1,8,3]),torch.tensor([1,2,4]))
        self.assertEqual(metric['unknown_n'],0);self.assertEqual(metric['unknown_correct_reject'],0)
        self.assertEqual(metric['known_correct'],1);self.assertEqual(metric['known_false_unknown'],1)
        self.assertEqual(metric['known_wrong_value'],1)


if __name__=='__main__':unittest.main()
