import unittest
from evaluation.calibrate_fact_reader import examples,metric,CAL_POS,CAL_NEG,TEST_POS,TEST_NEG
from evaluation.train_fact_reader import POS_TRAIN,NEG_TRAIN,POS_HELD,NEG_HELD


class CalibrationProtocolTests(unittest.TestCase):
    def test_exact_template_splits_disjoint(self):
        self.assertTrue(set(CAL_POS+CAL_NEG).isdisjoint(TEST_POS+TEST_NEG))
        self.assertTrue(set(CAL_POS+CAL_NEG+TEST_POS+TEST_NEG).isdisjoint(POS_TRAIN+NEG_TRAIN+POS_HELD+NEG_HELD))
        rows=examples(2,100,CAL_POS,CAL_NEG)
        self.assertEqual(sum(x['valid'] for x in rows),50)
        self.assertEqual(rows,examples(2,100,CAL_POS,CAL_NEG))

    def test_reject_all_not_perfect_precision(self):
        rows=[{'valid':True,'entity':0,'color':0,'pred_entity':0,'pred_color':0,
               'valid_score':.8,'entity_margin':.8,'color_margin':.8},
              {'valid':False,'entity':0,'color':0,'pred_entity':0,'pred_color':0,
               'valid_score':.8,'entity_margin':.8,'color_margin':.8}]
        empty=metric(rows,.9,.2)
        self.assertIsNone(empty['precision']);self.assertEqual(empty['accepted'],0)
        accepted=metric(rows,.7,.2)
        self.assertEqual(accepted['precision'],.5)
        self.assertEqual(accepted['wrong_accepted'],1)


if __name__=='__main__':unittest.main()
