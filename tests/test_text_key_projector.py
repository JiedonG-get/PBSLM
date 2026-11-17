import unittest
import torch
from models.text_key_projector import TextKeyProjector,key_alignment_loss
from evaluation.text_key_transfer import projection_matrix


class TextKeyProjectorTests(unittest.TestCase):
    def test_projection_only_gradients_and_identity_targets(self):
        p=TextKeyProjector(projection_matrix(1));raw=torch.randn(12,96)
        support=p(raw);query=p(raw+.01*torch.randn_like(raw))
        loss,parts=key_alignment_loss(support,query);loss.backward()
        self.assertEqual(sum(x.numel() for x in p.parameters()),1536)
        self.assertGreater(float(p.weight.grad.abs().sum()),0.)
        torch.testing.assert_close(support.norm(dim=-1),torch.ones(12),atol=1e-6,rtol=1e-6)
        self.assertTrue(torch.isfinite(parts['identity']))
        with self.assertRaises(ValueError):key_alignment_loss(support[:1],query[:1])


if __name__=='__main__':unittest.main()
