"""Text fact split/tokenization protocol tests; no training."""
import unittest
import torch
from evaluation.text_facts import TextFacts, TEMPLATES, ENTITIES, COLORS


class TextFactProtocolTests(unittest.TestCase):
    def test_split_and_token_contract(self):
        task = TextFacts('cpu')
        self.assertEqual(len(TEMPLATES),6)
        self.assertEqual(len(ENTITIES),16)
        self.assertEqual(len(COLORS),8)
        self.assertEqual(len(task.labels.unique()),8)
        self.assertGreater(task.vocab,8)
        encodings = [tuple(x.tolist()) for x in task.inputs.values()]
        self.assertEqual(len(set(encodings)), len(encodings))
        self.assertEqual(len(task.original_ids), len(set(task.original_ids)))
        self.assertTrue(task.token_inventory_evidence['transductive_vocab_and_padding'])
        for ids in task.inputs.values():
            self.assertEqual(tuple(ids.shape),(task.length,))
            self.assertTrue(bool((ids>=0).all() and (ids<task.vocab).all()))

    def test_random_episode_split_roles(self):
        task=TextFacts('cpu')
        gen=torch.Generator().manual_seed(12)
        sx,qx,hx,y,meta=task.episode(gen,'familiar')
        self.assertEqual(len(set(meta['entities'])),4)
        self.assertTrue(all(x<12 for x in meta['entities']))
        self.assertTrue(all(x in (0,1) for x in meta['support_templates']))
        self.assertTrue(all(x in (2,3) for x in meta['query_templates']))
        self.assertTrue(all(x in (4,5) for x in meta['held_templates']))
        self.assertEqual(len(y.unique()),4)
        _,_,_,_,novel=task.episode(gen,'novel')
        self.assertTrue(all(x>=12 for x in novel['entities']))


if __name__=='__main__': unittest.main()
