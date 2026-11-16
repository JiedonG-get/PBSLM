"""Consolidation isolation contracts, CPU random model, no research training run."""
import unittest
import torch
from models.model_v2 import PBSLMv2
from evaluation.text_facts import TextFacts,model_config
from evaluation.memory_consolidation import train_steps,protected_hash


class ConsolidationContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(33);self.task=TextFacts('cpu')
        self.model=PBSLMv2(**model_config(self.task)).eval()
        self.model.freeze_for_online()
        self.targets=self.task.labels[:4,None].expand(-1,2)

    def test_training_rejects_student_memory_access(self):
        with self.assertRaisesRegex(ValueError,'memory disabled'):
            train_steps(self.model,self.task,list(range(4)),self.targets,1,0)
        self.model.use_mem=False
        with torch.no_grad():self.model.mem[0].fill_(.1)
        with self.assertRaisesRegex(ValueError,'memory disabled'):
            train_steps(self.model,self.task,list(range(4)),self.targets,1,0)

    def test_one_step_only_updates_online_whitelist_without_m_or_c(self):
        self.model.use_mem=False
        before=protected_hash(self.model)
        parameters={name:p.detach().clone() for name,p in self.model.named_parameters() if p.requires_grad}
        train_steps(self.model,self.task,list(range(4)),self.targets,1,0)
        self.assertEqual(before,protected_hash(self.model))
        self.assertTrue(any(not torch.equal(parameters[n],p) for n,p in self.model.named_parameters() if n in parameters))
        self.assertTrue(all(mem.count_nonzero()==0 for mem in self.model.mem))


if __name__=='__main__':unittest.main()
