"""Persisted compact BPE/model contract; no training."""
import tempfile
from pathlib import Path
import unittest

import torch
from models.model_v2 import PBSLMv2
from evaluation.text_facts import TextFacts, model_config
from evaluation.text_checkpoint import save,load


class TextCheckpointTests(unittest.TestCase):
    def test_strict_roundtrip_and_clean_memory(self):
        task=TextFacts('cpu'); torch.manual_seed(44)
        model=PBSLMv2(**model_config(task)).eval()
        with torch.no_grad(): model.mem[0].fill_(.5)
        before=model.mem[0].clone()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'text.pt'
            manifest=save(path,model,task,{'seed':44},{'torch':torch.__version__})
            restored,t2,meta=load(path)
            self.assertEqual(manifest['sha256'],meta['sha256'])
            self.assertEqual(task.original_ids,t2.original_ids)
            self.assertTrue(torch.equal(model.mem[0],before))
            self.assertTrue(all(value.count_nonzero()==0 for value in restored.mem))
            for name,value in model.state_dict().items():
                if not name.startswith('mem.'):
                    self.assertTrue(torch.equal(value,restored.state_dict()[name]))
            with self.assertRaises(FileExistsError): save(path,model,task,{}, {})

    def test_corrupted_task_mapping_rejected(self):
        task=TextFacts('cpu'); model=PBSLMv2(**model_config(task)).eval()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'text.pt'; save(path,model,task,{}, {})
            ck=torch.load(path,weights_only=True)
            ck['task']['compact_to_bpe_ids'][0]=999
            torch.save(ck,path)
            with self.assertRaisesRegex(ValueError,'inventory'):
                load(path)

    def test_config_or_nonfinite_rejected(self):
        task=TextFacts('cpu'); model=PBSLMv2(**model_config(task)).eval()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'text.pt'; save(path,model,task,{}, {})
            ck=torch.load(path,weights_only=True); ck['model_config']['use_flow']=False; torch.save(ck,path)
            with self.assertRaisesRegex(ValueError,'configuration'):
                load(path)
            path2=Path(d)/'bad.pt'; save(path2,model,task,{}, {})
            ck=torch.load(path2,weights_only=True); ck['state_dict']['tok.weight'][0,0]=float('nan'); torch.save(ck,path2)
            with self.assertRaises(FloatingPointError): load(path2)

