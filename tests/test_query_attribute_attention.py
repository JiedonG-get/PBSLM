import unittest
from evaluation.query_attribute_attention_smoke import trial


class QueryAttributeAttentionTests(unittest.TestCase):
    def test_query_attributes_correction_and_distractors(self):
        for seed in (0,1,2):
            result=trial(seed)
            self.assertEqual(result['query_attribute_correct'],24)
            self.assertEqual(result['permuted_correct'],24)
            self.assertEqual(result['correction_correct'],24)
            self.assertTrue(result['untouched_value_records_bitwise_unchanged'])
            self.assertEqual(result['unrelated_duplicates_correct'],3)
            self.assertLess(result['object_only_correct'],24)

    def test_conflicting_records_remain_ambiguous(self):
        for seed in (0,1,2):
            result=trial(seed)
            # Both contradictory equally keyed records attract substantial mass.
            self.assertGreater(result['contradictory_equal_key_old_score'],.4)
            self.assertGreater(result['contradictory_equal_key_new_score'],.4)


if __name__=='__main__':unittest.main()
