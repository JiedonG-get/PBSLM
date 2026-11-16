import unittest
import torch
from models.model_v2 import PBSLMv2
from evaluation.text_facts import TextFacts,model_config
from evaluation.sequential_feature_consolidation import randomize_e_only


class RandomEAblationTests(unittest.TestCase):
    def test_randomization_changes_only_e_bank_and_mix(self):
        task=TextFacts('cpu');torch.manual_seed(3)
        model=PBSLMv2(**model_config(task)).eval();model.use_mem=False
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        randomize_e_only(model,task,777)
        changed=[name for name,value in model.state_dict().items() if not torch.equal(value,before[name])]
        self.assertTrue(changed)
        self.assertTrue(all(name=='e_mix' or name.startswith('ebank.') for name in changed))
        clone=PBSLMv2(**model_config(task)).eval();clone.load_state_dict(before)
        randomize_e_only(clone,task,777)
        self.assertTrue(all(torch.equal(model.state_dict()[name],value) for name,value in clone.state_dict().items()))


if __name__=='__main__':unittest.main()
