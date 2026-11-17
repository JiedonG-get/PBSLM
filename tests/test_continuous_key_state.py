import unittest
import torch
from models.continuous_key_state import ContinuousKeyState
from evaluation.continuous_key_state_control import make_episode,UNKNOWN


class ContinuousKeyTaskTests(unittest.TestCase):
    def test_fresh_keys_and_unknown_are_not_seen_flag_inputs(self):
        model=ContinuousKeyState(width=24).eval();generator=torch.Generator().manual_seed(6)
        with torch.no_grad():episode=make_episode(model,32,24,generator,'cpu',.01)
        self.assertEqual(tuple(episode['targets'].shape),(32,16))
        self.assertTrue(((episode['targets']==UNKNOWN).sum()>0))
        self.assertTrue(((episode['targets']!=UNKNOWN).sum()>0))
        self.assertNotIn('ids',episode)
        self.assertNotIn('seen_mask',episode)
        self.assertEqual(tuple(episode['state'].shape),(32,24))

    def test_corrections_change_truth_but_queries_do_not_mutate_state(self):
        model=ContinuousKeyState(width=24).eval();gen=torch.Generator().manual_seed(9)
        with torch.no_grad():
            result=make_episode(model,16,96,gen,'cpu',.05)
            self.assertGreater(result['correction_events'],0)
            self.assertTrue(torch.isfinite(result['state']).all())
            # The API exposes only one final state; prediction is a pure function.
            before=result['state'].clone();model.predict(before,result['keys'][:,0])
            self.assertTrue(torch.equal(before,result['state']))


if __name__=='__main__':unittest.main()
