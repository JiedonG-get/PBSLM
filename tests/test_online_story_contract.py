import unittest
from evaluation.online_story_contract import load_cases,input_view,score


class OnlineStoryContractTests(unittest.TestCase):
    def test_backend_view_excludes_labels_and_future(self):
        data=load_cases();case=data['cases'][0]
        view=input_view(case,1)
        self.assertEqual(len(view['events']),2)
        self.assertEqual(set(view),{'case_id','question','events'})
        self.assertTrue(all(set(event)=={'id','text'} for event in view['events']))
        self.assertNotIn('图书馆门口',str(view))
        with self.assertRaises(ValueError):input_view(case,-1)

    def test_synthetic_oracle_only_checks_scorer(self):
        data=load_cases()
        oracle=[{'case_id':c['id'],'event_id':e['id'],'answer':e['target_after'],'interrupt':e['interrupt']}
                 for c in data['cases'] for e in c['events']]
        result=score(data,oracle)
        self.assertEqual(result['events'],16);self.assertEqual(result['answer_exact_all_events'],1.)
        self.assertEqual(result['missed_required_interrupts'],0)
        self.assertEqual(result['false_interrupts'],0)
        with self.assertRaises(ValueError):score(data,oracle+[oracle[0]])

    def test_empty_predictions_not_counted_as_perfect_silence(self):
        result=score(load_cases(),[])
        self.assertEqual(result['coverage'],0)
        self.assertEqual(result['answer_exact_all_events'],0)
        self.assertEqual(result['missed_required_interrupts'],result['required_interrupt_events'])


if __name__=='__main__':unittest.main()
