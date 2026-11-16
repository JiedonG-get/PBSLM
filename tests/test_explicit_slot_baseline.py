import unittest
from evaluation.explicit_slot_baseline import baseline


def event(i,key,feedback,truths,correction=False):
    return {'event':i,'entity':key,'feedback':feedback,'correction':correction,
            'probes':[{'entity':k,'expected_latest_truth':v} for k,v in sorted(truths.items())]}


class ExplicitSlotBaselineTests(unittest.TestCase):
    def test_hidden_truth_never_replaces_wrong_feedback(self):
        rows=[event(0,0,'蓝',{0:'红'}),event(1,0,'红',{0:'红'},True)]
        result=baseline(rows,8)
        self.assertFalse(result['events'][0]['probes'][0]['correct'])
        self.assertEqual(result['events'][0]['probes'][0]['answer'],'蓝')
        self.assertEqual(result['corrective_feedback_post_accuracy'],1.)

    def test_eviction_abstains_and_counts_in_denominator(self):
        rows=[event(0,0,'红',{0:'红'}),event(1,1,'蓝',{0:'红',1:'蓝'}),
              event(2,2,'绿',{0:'红',1:'蓝',2:'绿'},True)]
        result=baseline(rows,2)
        last=result['events'][-1]
        self.assertTrue(last['probes'][0]['abstained'])
        self.assertEqual(last['correct'],2);self.assertEqual(last['n'],3)
        self.assertEqual(last['retained_slots'],[1,2])
        self.assertEqual(result['accepted_precision'],1.)


if __name__=='__main__':unittest.main()
