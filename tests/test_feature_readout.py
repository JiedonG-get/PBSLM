import copy
import unittest
import torch
from models.model_v2 import PBSLMv2
from models.feature_readout import FrozenFeatureReadout
from evaluation.text_facts import TextFacts,model_config
from evaluation.memory_consolidation import logits


class FrozenFeatureReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7);self.task=TextFacts('cpu')
        self.base=PBSLMv2(**model_config(self.task)).eval();self.base.use_mem=False

    def test_matched_readout_capacity_and_only_readout_gradients(self):
        models=[]
        for feature in ('e_key','token_mean'):
            torch.manual_seed(10)
            model=FrozenFeatureReadout(copy.deepcopy(self.base),feature)
            x=torch.stack([self.task.inputs[0,2],self.task.inputs[1,3]])
            features=model.features(x)
            self.assertEqual(tuple(features.shape),(2,self.base.e_n*self.base.d))
            self.assertFalse(features.requires_grad)
            scores=logits(model,x)
            torch.nn.functional.cross_entropy(scores,self.task.labels[:2]).backward()
            self.assertTrue(all(p.grad is None for p in model.base.parameters()))
            self.assertGreater(float(model.readout.weight.grad.abs().sum()),0.)
            models.append(model)
        self.assertTrue(torch.equal(models[0].readout.weight,models[1].readout.weight))
        self.assertEqual(sum(p.numel() for p in models[0].online_parameters()),sum(p.numel() for p in models[1].online_parameters()))

    def test_no_memory_access_and_explicit_mode_contract(self):
        model=FrozenFeatureReadout(self.base)
        self.base.use_mem=True
        with self.assertRaises(ValueError):model.features(self.task.inputs[0,2][None])
        self.base.use_mem=False
        with torch.no_grad():self.base.mem[0].fill_(1)
        with self.assertRaises(ValueError):model.features(self.task.inputs[0,2][None])
        with self.assertRaises(ValueError):model(self.task.inputs[0,2][None],m=2)


if __name__=='__main__':unittest.main()
