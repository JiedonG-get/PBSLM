import unittest
import random
import torch
from evaluation.train_fact_reader import example,metrics
from runtime.fact_reader import CandidateReader,encode,infer,tokenizer


class FactReaderProtocolTests(unittest.TestCase):
    def test_shapes_and_explicit_input_contract(self):
        tok,_=tokenizer();model=CandidateReader(tok.get_vocab_size()).eval()
        rows=[example(random.Random(i),positive=i%2==0)['text'] for i in range(4)]
        ids,lengths=encode(rows,tok,'cpu')
        with torch.no_grad():v,e,c=model(ids,lengths)
        self.assertEqual(tuple(v.shape),(4,2));self.assertEqual(tuple(e.shape),(4,16));self.assertEqual(tuple(c.shape),(4,8))
        with self.assertRaises(ValueError):encode([''],tok,'cpu')
        with self.assertRaises(ValueError):encode(['颜色'*100],tok,'cpu')
        predictions=infer(model,rows,tok,'cpu',threshold=1.)
        self.assertTrue(all(not x['accepted'] for x in predictions))

    def test_negative_accept_counts_as_error_even_correct_fields(self):
        rows=[{'truth':{'valid':0,'entity':1,'color':2},'prediction':{'accepted':True,'entity':1,'color':2}},
              {'truth':{'valid':1,'entity':1,'color':2},'prediction':{'accepted':True,'entity':1,'color':2}}]
        m=metrics(rows)
        self.assertEqual(m['accepted_exact_precision'],.5)
        self.assertEqual(m['negative_false_accept_rate'],1.)
        self.assertEqual(m['accepted_wrong_candidates'],1)


if __name__=='__main__':unittest.main()
