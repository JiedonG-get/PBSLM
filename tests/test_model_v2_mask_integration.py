"""O1.5 end-to-end integration tests for PBSLMv2's optional per-sample mask path.

Forward-only; no optimizer.step and no training.
"""
import unittest

import torch

from models.model_v2 import PBSLMv2


def make_model(use_sample_masks, dense=False):
    torch.manual_seed(0)
    return PBSLMv2(vocab=64, k=8, depth=1, d=32, heads=4, m_max=8,
                   act_x=0.2, e_n=2, e_depth=1, emo_dim=8, field_scale=0.0,
                   use_mem=True, trainable_target=True, simple_dec=False,
                   dense=dense, use_flow=True, direct_only=False,
                   use_sample_masks=use_sample_masks).eval()


def forward(model, ids):
    with torch.no_grad():
        return model(ids, 2, act=0.2, sample_noise=False)


class ModelV2MaskIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.ids = torch.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16],
        ])
        self.model = make_model(use_sample_masks=True)
        self.out = forward(self.model, self.ids)

    def test_masked_forward_fields_and_finiteness(self):
        o = self.out
        for k in ("chosen_a", "chosen_c", "active_a", "active_c",
                  "active_union_global", "mask_a", "mask_c"):
            self.assertIn(k, o)
        G = len(o["active_union_global"])
        self.assertEqual(tuple(o["mask_a"].shape), (G, 2))
        self.assertEqual(tuple(o["mask_c"].shape), (G, 2))
        self.assertEqual(tuple(o["z_a"].shape), (2, 2, 32))
        self.assertEqual(tuple(o["z_c"].shape), (2, 2, 32))
        self.assertTrue(torch.isfinite(o["z_a"]).all())
        self.assertTrue(torch.isfinite(o["z_c"]).all())
        self.assertTrue(torch.isfinite(o["I_a"]).all())
        self.assertTrue(torch.isfinite(o["I_c"]).all())

    def test_index_mask_mapping(self):
        o = self.out
        union = o["active_union_global"]
        for b in range(2):
            a = {union[g] for g in range(len(union)) if bool(o["mask_a"][g, b])}
            c = {union[g] for g in range(len(union)) if bool(o["mask_c"][g, b])}
            self.assertEqual(a, set(o["chosen_a"][b]))
            self.assertEqual(c, set(o["chosen_c"][b]))
        self.assertEqual(set(union), set(sum(o["chosen_a"], [])) | set(sum(o["chosen_c"], [])))
        self.assertEqual(union, sorted(union))

    def test_dense_uses_all_blocks_compat(self):
        dense = make_model(use_sample_masks=True, dense=True)
        out = forward(dense, self.ids)
        allb = list(range(dense.k))
        for b in range(2):
            self.assertEqual(out["chosen_a"][b], allb)
            self.assertEqual(out["chosen_c"][b], allb)
        self.assertEqual(out["active_union_global"], allb)
        self.assertTrue(bool(out["mask_a"].all()))
        self.assertTrue(bool(out["mask_c"].all()))

    def test_invalid_sparse_quota_raises(self):
        m = make_model(use_sample_masks=True)
        score = torch.zeros(2, m.k)
        for kA, kC in ((m.k + 1, 1), (1, m.k + 1), (0, 1), (1, 0)):
            with self.assertRaises(ValueError):
                m._select_masked(score, score, kA, kC)

    def test_independent_quota_allows_sum_exceed_k(self):
        m = make_model(use_sample_masks=True)
        score = torch.arange(m.k, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
        ia, ic = m._select_masked(score, score, m.k, 1)
        self.assertEqual(tuple(ia.shape), (2, m.k))
        self.assertEqual(tuple(ic.shape), (2, 1))

    def test_independent_topk_identical_partial_disjoint(self):
        m = make_model(use_sample_masks=True)
        B, k = 1, m.k
        # identical selections
        s = torch.tensor([[8, 7, 6, 5, 4, 3, 2, 1]], dtype=torch.float32)
        ia, ic = m._select_masked(s, s, 3, 3)
        self.assertEqual(set(ia[0].tolist()), set(ic[0].tolist()))
        # partial overlap: A {0,1,2}, C {1,2,3}
        sa = s
        sc = torch.tensor([[1, 8, 7, 6, 5, 4, 3, 2]], dtype=torch.float32)
        ia, ic = m._select_masked(sa, sc, 3, 3)
        self.assertEqual(set(ia[0].tolist()), {0, 1, 2})
        self.assertEqual(set(ic[0].tolist()), {1, 2, 3})
        # disjoint: A {0,1,2}, C {3,4,5}
        sc2 = torch.tensor([[0, 0, 0, 8, 7, 6, 5, 4]], dtype=torch.float32)
        ia, ic = m._select_masked(sa, sc2, 3, 3)
        self.assertTrue(set(ia[0].tolist()).isdisjoint(set(ic[0].tolist())))

    def test_independent_routing_per_sample_scores(self):
        m = make_model(use_sample_masks=True)
        sa = torch.tensor([[8, 7, 6, 5, 4, 3, 2, 1],
                           [0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.float32)
        sc = torch.tensor([[8, 7, 6, 5, 4, 3, 2, 1],
                           [7, 6, 5, 4, 3, 2, 1, 0]], dtype=torch.float32)
        ia, ic = m._select_masked(sa, sc, 1, 2)
        self.assertEqual(ia[0].tolist(), [0])
        self.assertEqual(ia[1].tolist(), [7])
        self.assertEqual(ic[0].tolist(), [0, 1])
        self.assertEqual(ic[1].tolist(), [0, 1])

    def test_overlap_masks_union_and_side_masks(self):
        model = make_model(use_sample_masks=True)
        sa = torch.tensor([[9, 8, 7, 6, 5, 4, 3, 2],
                           [9, 8, 7, 6, 5, 4, 3, 2]], dtype=torch.float32)
        sc = torch.tensor([[8, 9, 7, 6, 5, 4, 3, 2],
                           [0, 0, 9, 8, 7, 6, 5, 4]], dtype=torch.float32)
        old_forward_a, old_forward_c = model.role_a.forward, model.role_c.forward
        model.role_a.forward = lambda h: sa
        model.role_c.forward = lambda h: sc
        comm_masks, flow_masks = [], []
        old_comm = model.bank.comm
        old_flow_forward = model.flow.forward

        def comm_wrap(Y, g, idx=None, mask=None, trace=None):
            comm_masks.append(None if mask is None else mask.clone())
            return old_comm(Y, g, idx=idx, mask=mask, trace=trace)

        def flow_wrap(delta, mask=None, trace=None):
            flow_masks.append(None if mask is None else mask.clone())
            return old_flow_forward(delta, mask=mask, trace=trace)

        model.bank.comm = comm_wrap
        model.flow.forward = flow_wrap
        try:
            out = forward(model, self.ids)
        finally:
            model.role_a.forward, model.role_c.forward = old_forward_a, old_forward_c
            model.bank.comm = old_comm
            model.flow.forward = old_flow_forward
        self.assertEqual(len(comm_masks), 2)
        self.assertEqual(len(flow_masks), 2)
        self.assertTrue(torch.equal(comm_masks[0], out["mask_a"]))
        self.assertTrue(torch.equal(comm_masks[1], out["mask_c"]))
        self.assertTrue(torch.equal(flow_masks[0], out["mask_a"]))
        self.assertTrue(torch.equal(flow_masks[1], out["mask_c"]))
        overlap = set(out["chosen_a"][0]) & set(out["chosen_c"][0])
        self.assertTrue(overlap)
        for blk in overlap:
            row = out["active_union_global"].index(blk)
            self.assertTrue(bool(out["mask_a"][row, 0]))
            self.assertTrue(bool(out["mask_c"][row, 0]))

    def test_per_sample_masks_can_differ(self):
        o = self.out
        self.assertFalse(torch.equal(o["mask_a"][:, 0], o["mask_a"][:, 1])
                         and torch.equal(o["mask_c"][:, 0], o["mask_c"][:, 1]))

    def test_cross_sample_isolation(self):
        ids2 = self.ids.clone()
        ids2[1] = torch.tensor([17, 18, 19, 20, 21, 22, 23, 24])
        o2 = forward(self.model, ids2)
        # Sample 0's routing/output must not depend on sample 1's input.
        self.assertEqual(self.out["chosen_a"][0], o2["chosen_a"][0])
        self.assertEqual(self.out["chosen_c"][0], o2["chosen_c"][0])
        for key in ("z_a", "z_c", "I_a", "I_c", "h_a"):
            torch.testing.assert_close(self.out[key][0], o2[key][0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(self.out["z_c"][0], o2["z_c"][0], atol=1e-6, rtol=1e-5)

    def test_legacy_path_remains_default_and_masked_fields_absent(self):
        legacy = make_model(use_sample_masks=False)
        out = forward(legacy, self.ids)
        for k in ("mask_a", "mask_c", "chosen_a", "chosen_c",
                  "active_a", "active_c", "active_union_global"):
            self.assertNotIn(k, out)
        self.assertIn("active", out)
        self.assertIn("chosen", out)

    def test_parameter_and_state_dict_contract(self):
        masked = make_model(use_sample_masks=True)
        legacy = make_model(use_sample_masks=False)
        self.assertEqual(set(masked.state_dict()), set(legacy.state_dict()))
        self.assertEqual(sum(p.numel() for p in masked.parameters()),
                         sum(p.numel() for p in legacy.parameters()))
        # Old checkpoints with the same architecture can load.
        legacy.load_state_dict(masked.state_dict())
        out = forward(legacy, self.ids)
        self.assertTrue(torch.isfinite(out["z_a"]).all())

    def test_invalid_block_nan_injected_at_comm_does_not_pollute(self):
        baseline = forward(self.model, self.ids)
        orig_comm = self.model.bank.comm

        def comm_with_invalid_nan(Y, g, idx=None, mask=None, trace=None):
            Y = Y.clone()
            if mask is not None:
                Y[~mask] = float("nan")
            return orig_comm(Y, g, idx=idx, mask=mask, trace=trace)

        self.model.bank.comm = comm_with_invalid_nan
        try:
            out = forward(self.model, self.ids)
        finally:
            self.model.bank.comm = orig_comm
        self.assertTrue(torch.isfinite(out["z_a"]).all())
        self.assertTrue(torch.isfinite(out["z_c"]).all())
        torch.testing.assert_close(out["z_a"], baseline["z_a"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(out["z_c"], baseline["z_c"], atol=1e-6, rtol=1e-5)

    def test_shared_input_nan_raises_before_block_forward(self):
        orig_bf = self.model.bank.block_forward

        def bad_block_forward(x, idx):
            x = x.clone()
            x[0, 0, 0] = float("nan")
            return orig_bf(x, idx)

        self.model.bank.block_forward = bad_block_forward
        try:
            with self.assertRaises(FloatingPointError):
                forward(self.model, self.ids)
        finally:
            self.model.bank.block_forward = orig_bf

    def test_legacy_and_masked_use_same_output_head_contract(self):
        # O1 out_a/out_c split must remain; masked mode must not change them.
        o = self.out
        self.assertIn("z_a", o)
        self.assertIn("z_c", o)
        self.assertNotEqual(id(self.model.out_a), id(self.model.out_c))


if __name__ == "__main__":
    unittest.main()
