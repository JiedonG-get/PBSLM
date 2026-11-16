import unittest
import random
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from evaluation.train_fact_reader import example,metrics
from runtime.fact_reader import CandidateReader,encode,infer,tokenizer,load_reader
from evaluation.text_facts import ENTITIES,COLORS


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

    def test_policy_and_nonfinite_logits_rejected(self):
        tok,_=tokenizer();model=CandidateReader(tok.get_vocab_size()).eval()
        for threshold,margin in [(float('nan'),.2),(.9,-.1),(1.1,.2),(.9,float('inf')),(True,.2)]:
            with self.assertRaises(ValueError):infer(model,['甲号设备是红色。'],tok,'cpu',threshold,margin)
        with patch.object(model,'forward',return_value=(torch.full((1,2),float('nan')),torch.zeros(1,16),torch.zeros(1,8))):
            with self.assertRaises(FloatingPointError):infer(model,['甲号设备是红色。'],tok,'cpu')
        with self.assertRaises(ValueError):encode([],tok,'cpu')

    def test_load_metadata_validation(self):
        tok,digest=tokenizer();model=CandidateReader(tok.get_vocab_size()).eval()
        original={'format':'pbslm.fact_reader.v1','config':{'vocab':tok.get_vocab_size(),'embedding':32,'hidden':48},
                  'tokenizer_sha256':digest,'entities':ENTITIES,'colors':COLORS,
                  'state_dict':model.state_dict(),'threshold':.9,'margin':.2}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reader.pt';torch.save(original,path)
            load_reader(path)
            bad={**original,'threshold':float('nan')};torch.save(bad,path)
            with self.assertRaises(ValueError):load_reader(path)
            bad={**original,'config':{**original['config'],'vocab':4}};torch.save(bad,path)
            with self.assertRaises(ValueError):load_reader(path)

    def test_negative_accept_counts_as_error_even_correct_fields(self):
        rows=[{'truth':{'valid':0,'entity':1,'color':2},'prediction':{'accepted':True,'entity':1,'color':2}},
              {'truth':{'valid':1,'entity':1,'color':2},'prediction':{'accepted':True,'entity':1,'color':2}}]
        m=metrics(rows)
        self.assertEqual(m['accepted_exact_precision'],.5)
        self.assertEqual(m['negative_false_accept_rate'],1.)
        self.assertEqual(m['accepted_wrong_candidates'],1)


if __name__=='__main__':unittest.main()
