"""Continuous-vector input compatibility for PBSLMv2."""
import unittest

import torch

from models.model_v2 import PBSLMv2


def make_model(use_sample_masks):
    torch.manual_seed(7)
    return PBSLMv2(vocab=32, k=8, depth=1, d=32, heads=4, m_max=4,
                   act_x=0.5, e_n=2, e_depth=1, field_scale=0.0,
                   use_sample_masks=use_sample_masks).eval()


class ModelV2InputsEmbedsTests(unittest.TestCase):
    def setUp(self):
        self.ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    def test_embedding_path_matches_token_path(self):
        for use_sample_masks in (False, True):
            with self.subTest(use_sample_masks=use_sample_masks):
                model = make_model(use_sample_masks)
                embeddings = model.token_emb(self.ids)
                with torch.no_grad():
                    token_output = model(self.ids, 2, act=0.5, sample_noise=False)
                    embed_output = model(None, 2, inputs_embeds=embeddings, act=0.5,
                                         sample_noise=False)
                for name in ("z_a", "z_c", "h_a", "I_a", "I_c"):
                    torch.testing.assert_close(embed_output[name], token_output[name])
                self.assertEqual(token_output.keys(), embed_output.keys())

    def test_embedding_path_propagates_gradients(self):
        model = make_model(use_sample_masks=True)
        embeddings = model.token_emb(self.ids).detach().requires_grad_()
        output = model(inputs_embeds=embeddings, m=2, act=0.5, sample_noise=False)
        weights = torch.arange(1, 33, dtype=output["z_a"].dtype).view(1, 1, -1)
        (output["z_a"] * weights).sum().backward()
        self.assertIsNotNone(embeddings.grad)
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertGreater(float(embeddings.grad.norm()), 0.0)

    def test_embedding_argument_validation(self):
        model = make_model(use_sample_masks=False)
        embeddings = model.token_emb(self.ids)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model(self.ids, 2, inputs_embeds=embeddings, sample_noise=False)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model(m=2, sample_noise=False)
        with self.assertRaisesRegex(ValueError, "shape"):
            model(inputs_embeds=torch.zeros(2, 4, 31), m=2, sample_noise=False)
        with self.assertRaisesRegex(TypeError, "floating-point"):
            model(inputs_embeds=torch.ones(2, 4, 32, dtype=torch.long),
                  m=2, sample_noise=False)
        with self.assertRaisesRegex(ValueError, "m is required"):
            model(inputs_embeds=embeddings, sample_noise=False)


if __name__ == "__main__":
    unittest.main()
