"""Task protocol checks; small CPU construction, no optimizer or training."""
import unittest
import torch
from evaluation.episodic_memory import episode, fill_memory, held_eval
from models.model_v2 import PBSLMv2


class EpisodicProtocolTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(9)
        return PBSLMv2(vocab=48, k=8, depth=1, d=16, heads=4, m_max=2,
                        e_n=2, act_x=.5, use_sample_masks=True,
                        trainable_target=False, field_scale=0).eval()

    def test_episode_pairs_are_unique_and_fresh(self):
        g = torch.Generator().manual_seed(13)
        x, y = episode(g, 4, 'cpu')
        x2, y2 = episode(g, 4, 'cpu')
        self.assertEqual(len(x[:, 0].unique()), 4)
        self.assertEqual(len(y.unique()), 4)
        self.assertFalse(torch.equal(x, x2) and torch.equal(y, y2))

    def test_interventions_do_not_update_parameters(self):
        m = self.model()
        before = {n: p.detach().clone() for n, p in m.named_parameters() if not n.startswith('mem.')}
        x, y = episode(torch.Generator().manual_seed(11), 4, 'cpu')
        fill_memory(m, x, y, 'normal')
        self.assertTrue(any(p.norm() > 0 for p in m.mem))
        normal = [p.detach().clone() for p in m.mem]
        fill_memory(m, x, y, 'wrong_values')
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(normal, m.mem)))
        fill_memory(m, x, y, 'zero')
        self.assertTrue(all(p.count_nonzero() == 0 for p in m.mem))
        for name, value in before.items():
            self.assertTrue(torch.equal(value, dict(m.named_parameters())[name]))

    def test_held_evaluation_checks_c_and_weights(self):
        out = held_eval(self.model(), 2, 4, 123, 'cpu')
        self.assertTrue(out['parameter_hash_unchanged'])
        self.assertTrue(out['c_bitwise_stable'])
        self.assertEqual(out['arms']['normal']['targets'], 8)


if __name__ == '__main__':
    unittest.main()
