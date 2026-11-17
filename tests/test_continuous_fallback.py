import unittest
import torch
from evaluation.continuous_key_state_control import combine_state_fallback,UNKNOWN


class ContinuousFallbackTests(unittest.TestCase):
    def test_only_explicit_unknown_state_queries_trigger_fallback(self):
        state=torch.tensor([1,UNKNOWN,4,UNKNOWN]);dictionary=torch.tensor([7,2,3,UNKNOWN])
        hybrid=combine_state_fallback(state,dictionary)
        self.assertEqual(hybrid.tolist(),[1,2,4,UNKNOWN])
        with self.assertRaises(ValueError):combine_state_fallback(torch.zeros(2),torch.zeros(1))


if __name__=='__main__':unittest.main()
