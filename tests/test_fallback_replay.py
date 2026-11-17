import unittest
from collections import OrderedDict
from evaluation.working_state_fallback_replay import observed_final,apply_fallback


class FallbackReplayTests(unittest.TestCase):
    def test_observed_write_ledger_matches_only_retained_truth(self):
        ledgers,truth=observed_final(0,0,4,steps=5,batch=2)
        for b,ledger in enumerate(ledgers):
            self.assertEqual(len(ledger),4);self.assertIn(0,ledger)
            for entity,value in ledger.items():self.assertEqual(value,int(truth[b,entity]))

    def test_mismatched_streams_rejected(self):
        with self.assertRaises(ValueError):apply_fallback([[0]*8],[],list(range(8)))

    def test_fallback_answers_from_ledger_and_miss_abstains(self):
        route=[0,0,2,3,4,5,6,7]
        answers,cost=apply_fallback([[7]*8],[OrderedDict([(0,3)])],route)
        self.assertEqual(answers[0][:3],[3,None,7])
        self.assertEqual(cost,{'queries':2,'hits':1,'misses':1})


if __name__=='__main__':unittest.main()
