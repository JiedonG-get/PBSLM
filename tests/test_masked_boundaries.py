"""Nonfinite values produced after InfoFlow must fail at the masked model boundary."""
import unittest
from unittest.mock import patch
import torch
from tests.test_model_v2_mask_integration import make_model, forward


class MaskedBoundaryTests(unittest.TestCase):
    def test_outer_proj_out_error(self):
        m = make_model(True)
        ids = torch.arange(16).reshape(2, 8)
        original = m.flow.proj_out.forward

        def project(value):
            out = original(value)
            # InfoFlow consensus is 3D; model's outer per-block projection is 4D.
            return torch.full_like(out, float('nan')) if value.dim() == 4 else out

        with patch.object(m.flow.proj_out, 'forward', side_effect=project):
            with self.assertRaisesRegex(FloatingPointError, 'PBSLMv2.projected_a'):
                forward(m, ids)

    def test_decoder_output_error(self):
        m = make_model(True)
        ids = torch.arange(16).reshape(2, 8)
        with patch.object(m.out_c, 'forward', side_effect=lambda h: torch.full_like(h, float('inf'))):
            with self.assertRaisesRegex(FloatingPointError, 'PBSLMv2.z_c'):
                forward(m, ids)

    def test_no_flow_agreement_selected_is_one(self):
        m = make_model(True)
        m.use_flow = False
        out = forward(m, torch.arange(16).reshape(2, 8))
        torch.testing.assert_close(out['I_a'], torch.ones_like(out['I_a']))
        torch.testing.assert_close(out['I_c'], torch.ones_like(out['I_c']))


if __name__ == '__main__':
    unittest.main()
