import tempfile
from pathlib import Path
import unittest
import torch
from models.slotted_working_state import SlottedWorkingState
from evaluation.working_state_checkpoint import load_working_state


class WorkingStateCheckpointTests(unittest.TestCase):
    def payload(self):
        torch.manual_seed(24);model=SlottedWorkingState().eval()
        return model,{'format':'pbslm.symbolic_working_state.v1','architecture':'learned','seed':24,'updates':600,
                       'config':{'entities':8,'values':8,'state_floats':96},'state_dict':model.state_dict()}

    def test_roundtrip_exact_prediction_and_frozen_weights(self):
        model,payload=self.payload()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt';torch.save(payload,path)
            loaded,metadata=load_working_state(path)
            entity=torch.tensor([1,3]);value=torch.tensor([2,4])
            with torch.no_grad():
                state=model.observe(model.initial(2),entity,value)
                restored=loaded.observe(loaded.initial(2),entity,value)
                self.assertTrue(torch.equal(state,restored))
                self.assertTrue(torch.equal(model.query(state,entity),loaded.query(restored,entity)))
            self.assertEqual(len(metadata['file_sha256']),64)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))

    def test_malformed_contracts_rejected(self):
        _,payload=self.payload()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.pt'
            torch.save({**payload,'config':{'entities':9}},path)
            with self.assertRaises(ValueError):load_working_state(path)
            payload['state_dict']['keys'][0,0]=float('nan');torch.save(payload,path)
            with self.assertRaises(ValueError):load_working_state(path)


if __name__=='__main__':unittest.main()
