import unittest
from evaluation.collision_guard_control import guarded_counts


class CollisionGuardTests(unittest.TestCase):
    def test_decision_independent_of_answers(self):
        route=[0,0,2,3,4,5,6,7]
        a=guarded_counts([[0]*8],[[0]*8],route)
        b=guarded_counts([[7]*8],[[0]*8],route)
        self.assertEqual(a['flagged_entities'],[0,1])
        self.assertEqual(a['accepted'],b['accepted']);self.assertEqual(a['deferred'],2)
        self.assertEqual(b['accepted_errors'],6)
        self.assertEqual(b['deferred_original_errors'],2)

    def test_mismatched_batch_rejected(self):
        with self.assertRaises(ValueError):guarded_counts([[0]*8],[],list(range(8)))

    def test_no_collision_not_no_error(self):
        result=guarded_counts([[1]*8],[[0]*8],list(range(8)))
        self.assertEqual(result['accepted_errors'],8);self.assertEqual(result['deferred'],0)


if __name__=='__main__':unittest.main()
