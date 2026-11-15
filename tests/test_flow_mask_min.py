"""Minimal P0-11/P0-12 implementation tests for model.py mask paths.

Forward / diagnostic backward only; no optimizer.step and no training.
"""
import math
import unittest

import torch

from models.model import BlockBank, FlowStep, InfoFlow


def make_mask(G, B, dtype=torch.bool):
    # Different valid supports per sample; non-contiguous physical IDs in callers.
    m = torch.zeros(G, B, dtype=dtype)
    for b in range(B):
        start = b % G
        m[start, b] = True
        m[(start + 2) % G, b] = True
    return m


def legacy_comm(bank, Y, g):
    w = g[:, None, None, None]
    tot = (w * Y).sum(0)
    others = tot.unsqueeze(0) - w * Y
    r = bank.comm2(torch.nn.functional.gelu(bank.comm1(others)))
    beta = torch.sigmoid(bank.comm_beta[: Y.shape[0]])[:, None, None, None]
    return Y + beta * w * r


def legacy_flow_step(layer, h):
    q, k, v = layer.q(h), layer.k(h), layer.v(h)
    att = torch.softmax(torch.einsum("ibtd,jbtd->btij", q, k) / math.sqrt(q.shape[-1]), -1)
    msg = torch.einsum("btij,jbtd->ibtd", att, v)
    return layer.ln(h + layer.o(msg))


def legacy_info_flow(flow, delta):
    h = flow.proj_in(delta)
    for layer in flow.layers:
        h = layer(h)
    mean = h.mean(0)
    scale = h.norm(dim=-1).mean(0) + 1e-6
    agree = 1 - ((h - mean.unsqueeze(0)).norm(dim=-1) / scale.unsqueeze(0)).clamp(0, 2)
    cons = flow.proj_out(mean)
    return h, agree, cons


class CommTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.bank = BlockBank(nb=6, depth=1, d=8, heads=2)
        with torch.no_grad():
            self.bank.comm_beta.copy_(torch.tensor([-3., -2., -1., 0., 1., 2.]))
            self.bank.comm2.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4) * 0.01)
            self.bank.comm2.bias.copy_(torch.arange(8, dtype=torch.float32) * 0.01)
        self.Y = torch.randn(3, 2, 4, 8)
        self.g = torch.tensor([0.2, 0.7, 1.1])
        self.idx = torch.tensor([1, 3, 5])
        self.mask = make_mask(3, 2)

    def test_legacy_equivalence_with_contiguous_all_valid(self):
        Y = self.Y
        g = self.g
        old = legacy_comm(self.bank, Y, g)
        new = self.bank.comm(Y, g, idx=torch.arange(3), mask=torch.ones(3, 2, dtype=torch.bool))
        torch.testing.assert_close(old, new)

    def test_physical_id_and_mask(self):
        Y = self.Y
        mask = self.mask
        g = self.g
        idx = self.idx
        out = self.bank.comm(Y, g, idx=idx, mask=mask)
        # Manual reference preserving w and physical-id beta.
        w = g[:, None, None, None]
        m4 = mask[:, :, None, None]
        valid = torch.where(m4, Y, torch.zeros_like(Y))
        weighted = w * valid
        tot = weighted.sum(0)
        others = torch.where(m4, tot.unsqueeze(0) - weighted, torch.zeros_like(Y))
        r = self.bank.comm2(torch.nn.functional.gelu(self.bank.comm1(others)))
        beta = torch.sigmoid(self.bank.comm_beta[idx])[:, None, None, None]
        expected = torch.where(m4, valid + beta * w * r, torch.zeros_like(Y))
        torch.testing.assert_close(out, expected)
        # With distinct comm_beta, the wrong contiguous slice must produce a different result.
        beta_wrong = torch.sigmoid(self.bank.comm_beta[:3])[:, None, None, None]
        wrong = torch.where(m4, valid + beta_wrong * w * r, torch.zeros_like(Y))
        self.assertFalse(torch.allclose(out, wrong))
        self.assertTrue(torch.equal(out[~mask], torch.zeros_like(out[~mask])))

    def test_bad_parameter_combination(self):
        with self.assertRaises(ValueError):
            self.bank.comm(self.Y, self.g, idx=self.idx, mask=None)
        with self.assertRaises(ValueError):
            self.bank.comm(self.Y, self.g, idx=None, mask=self.mask)
        with self.assertRaises(ValueError):
            self.bank.comm(self.Y, self.g, idx=self.idx, mask=torch.ones(3, 3, dtype=torch.bool))

    def test_invalid_nan_isolated_and_valid_nan_raises(self):
        Y = self.Y.clone()
        invalid = ~self.mask
        Y[invalid] = float("nan")
        out = self.bank.comm(Y, self.g, idx=self.idx, mask=self.mask)
        self.assertTrue(torch.isfinite(out[self.mask]).all())
        Y2 = self.Y.clone()
        pos = self.mask.nonzero()[0]
        Y2[pos[0], pos[1]] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.bank.comm(Y2, self.g, idx=self.idx, mask=self.mask)

    def test_m1_per_sample_equivalence(self):
        full_trace = {}
        out = self.bank.comm(self.Y, self.g, idx=self.idx, mask=self.mask, trace=full_trace)
        errs = {}
        def check(name, a, b):
            torch.testing.assert_close(a, b)
            if a.dtype == torch.bool or b.dtype == torch.bool:
                errs[name] = max(errs.get(name, 0.0), 0.0 if torch.equal(a, b) else 1.0)
            else:
                errs[name] = max(errs.get(name, 0.0), (a - b).abs().max().item())
        for b in range(self.Y.shape[1]):
            pt = {}
            ob = self.bank.comm(self.Y[:, b:b+1], self.g, idx=self.idx,
                                mask=self.mask[:, b:b+1], trace=pt)
            check("out", out[:, b], ob[:, 0])
            for k in ("valid_Y", "weighted_Y", "others", "r", "update", "out"):
                check(k, full_trace[k][:, b], pt[k][:, 0])
            check("tot", full_trace["tot"][b], pt["tot"][0])
            check("beta", full_trace["beta"], pt["beta"])
        print(f"\n[comm M1] max_err={max(errs.values()):.3e}")

    def test_batch_one(self):
        out = self.bank.comm(self.Y[:, :1], self.g, idx=self.idx, mask=self.mask[:, :1])
        self.assertEqual(tuple(out.shape), (3, 1, 4, 8))

    def test_block_forward_input_nan_raises(self):
        x = torch.randn(2, 4, 8)
        x[0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.bank.block_forward(x, torch.tensor([0, 2]))

    def test_physical_id_validation(self):
        Y, g, mask = self.Y, self.g, self.mask
        with self.assertRaises(ValueError):
            self.bank.comm(Y, g, idx=self.idx.reshape(1, 3), mask=mask)
        with self.assertRaises(ValueError):
            self.bank.comm(Y, g, idx=torch.tensor([1, 3, 9]), mask=mask)
        with self.assertRaises(ValueError):
            self.bank.comm(Y, g, idx=torch.tensor([1, 3, -1]), mask=mask)
        with self.assertRaises(ValueError):
            self.bank.comm(Y, g, idx=torch.tensor([1, 3, 3]), mask=mask)
        with self.assertRaises(ValueError):
            self.bank.comm(Y, g.reshape(1, 3), idx=self.idx, mask=mask)

    def test_invalid_nan_backward_finite(self):
        Y = self.Y.clone()
        Y[~self.mask] = float("nan")
        Y = Y.requires_grad_(True)
        out = self.bank.comm(Y, self.g, idx=self.idx, mask=self.mask)
        out[self.mask].sum().backward()
        self.assertTrue(torch.isfinite(Y.grad[self.mask]).all())
        for p in [*self.bank.comm1.parameters(), *self.bank.comm2.parameters(), self.bank.comm_beta]:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())


class FlowStepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.layer = FlowStep(8)
        self.G, self.B, self.T = 3, 2, 4
        self.mask = make_mask(self.G, self.B)
        self.h = torch.randn(self.G, self.B, self.T, 8)

    def test_legacy_equivalence(self):
        self.assertTrue(torch.allclose(self.layer(self.h), legacy_flow_step(self.layer, self.h)))

    def test_shapes_and_masked_zero(self):
        tr = {}
        out = self.layer(self.h, self.mask, trace=tr)
        self.assertEqual(tuple(out.shape), (self.G, self.B, self.T, 8))
        self.assertTrue(torch.equal(out[~self.mask], torch.zeros_like(out[~self.mask])))
        self.assertEqual(tuple(tr["logits"].shape), (self.B, self.T, self.G, self.G))
        self.assertEqual(tuple(tr["Q"].shape), (self.B, self.T, self.G, 8))
        self.assertEqual(tuple(tr["key_mask"].shape), (self.B, 1, 1, self.G))

    def test_m1_per_sample_equivalence(self):
        ft = {}
        out = self.layer(self.h, self.mask, trace=ft)
        errs = {}
        def check(name, a, b):
            torch.testing.assert_close(a, b)
            if a.dtype == torch.bool or b.dtype == torch.bool:
                errs[name] = max(errs.get(name, 0.0), 0.0 if torch.equal(a, b) else 1.0)
            else:
                errs[name] = max(errs.get(name, 0.0), (a - b).abs().max().item())
        for b in range(self.B):
            pt = {}
            m1 = self.layer(self.h[:, b:b+1], self.mask[:, b:b+1], trace=pt)
            check("out", out[:, b], m1[:, 0])
            for k in ("h_in", "q", "k", "v", "msg", "out"):
                check(k, ft[k][:, b], pt[k][:, 0])
            for k in ("Q", "K", "V", "logits", "att", "key_mask"):
                check(k, ft[k][b], pt[k][0])
            check("count_b", ft["count_b"][b], pt["count_b"][0])
        print(f"\n[FlowStep M1] max_err={max(errs.values()):.3e}")

    def test_batch_one(self):
        out = self.layer(self.h[:, :1], self.mask[:, :1])
        self.assertEqual(tuple(out.shape), (self.G, 1, self.T, 8))

    def test_empty_mask_raises(self):
        with self.assertRaises(ValueError):
            self.layer(self.h, torch.zeros(self.G, self.B, dtype=torch.bool))

    def test_per_sample_empty_mask_raises(self):
        mask = self.mask.clone()
        mask[:, 0] = False
        with self.assertRaises(ValueError):
            self.layer(self.h, mask)

    def test_invalid_nan_and_valid_nan(self):
        h = self.h.clone()
        h[~self.mask] = float("nan")
        out = self.layer(h, self.mask)
        self.assertTrue(torch.isfinite(out[self.mask]).all())
        h2 = self.h.clone()
        pos = self.mask.nonzero()[0]
        h2[pos[0], pos[1]] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.layer(h2, self.mask)

    def test_backward_finite(self):
        h = self.h.clone().requires_grad_(True)
        out = self.layer(h, self.mask)
        loss = out[self.mask].sum()
        loss.backward()
        self.assertTrue(torch.isfinite(h.grad[self.mask]).all())
        for p in self.layer.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_invalid_nan_backward_finite(self):
        h = self.h.clone()
        h[~self.mask] = float("nan")
        h = h.requires_grad_(True)
        out = self.layer(h, self.mask)
        out[self.mask].sum().backward()
        self.assertTrue(torch.isfinite(h.grad[self.mask]).all())
        for p in self.layer.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())


class InfoFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.flow = InfoFlow(d=8, fdim=8, steps=2)
        self.G, self.B, self.T = 3, 2, 4
        self.mask = make_mask(self.G, self.B)
        self.delta = torch.randn(self.G, self.B, self.T, 8)

    def test_legacy_equivalence(self):
        old = legacy_info_flow(self.flow, self.delta)
        new = self.flow(self.delta)
        for a, b in zip(old, new):
            torch.testing.assert_close(a, b)

    def test_masked_stats_and_outputs(self):
        tr = {}
        h, agree, cons = self.flow(self.delta, self.mask, trace=tr)
        self.assertEqual(tuple(h.shape), (self.G, self.B, self.T, 8))
        self.assertEqual(tuple(agree.shape), (self.G, self.B, self.T))
        self.assertEqual(tuple(cons.shape), (self.B, self.T, 8))
        self.assertTrue(torch.equal(agree[~self.mask], torch.zeros_like(agree[~self.mask])))
        self.assertTrue(torch.isfinite(h[self.mask]).all())
        self.assertTrue(torch.isfinite(agree[self.mask]).all())
        self.assertTrue(torch.isfinite(cons).all())

    def test_batch_one(self):
        h, agree, cons = self.flow(self.delta[:, :1], self.mask[:, :1])
        self.assertEqual(tuple(h.shape), (self.G, 1, self.T, 8))
        self.assertEqual(tuple(agree.shape), (self.G, 1, self.T))
        self.assertEqual(tuple(cons.shape), (1, self.T, 8))

    def test_empty_mask_raises(self):
        with self.assertRaises(ValueError):
            self.flow(self.delta, torch.zeros(self.G, self.B, dtype=torch.bool))

    def test_per_sample_empty_mask_raises(self):
        mask = self.mask.clone()
        mask[:, 0] = False
        with self.assertRaises(ValueError):
            self.flow(self.delta, mask)

    def test_invalid_nan_and_valid_nan(self):
        d = self.delta.clone()
        d[~self.mask] = float("nan")
        h, agree, cons = self.flow(d, self.mask)
        self.assertTrue(torch.isfinite(h[self.mask]).all())
        d2 = self.delta.clone()
        pos = self.mask.nonzero()[0]
        d2[pos[0], pos[1]] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.flow(d2, self.mask)

    def test_m1_per_sample_equivalence(self):
        ft = {}
        h, agree, cons = self.flow(self.delta, self.mask, trace=ft)
        errs = {}
        def check(name, a, b):
            torch.testing.assert_close(a, b)
            if a.dtype == torch.bool or b.dtype == torch.bool:
                errs[name] = max(errs.get(name, 0.0), 0.0 if torch.equal(a, b) else 1.0)
            else:
                errs[name] = max(errs.get(name, 0.0), (a - b).abs().max().item())
        for b in range(self.B):
            pt = {}
            h1, a1, c1 = self.flow(self.delta[:, b:b+1], self.mask[:, b:b+1], trace=pt)
            check("h_out", h[:, b], h1[:, 0])
            check("agree_out", agree[:, b], a1[:, 0])
            check("cons_out", cons[b], c1[0])
            for k in ("proj_in", "h", "agree"):
                check(k, ft[k][:, b], pt[k][:, 0])
            for k in ("mean", "scale", "cons"):
                check(k, ft[k][b], pt[k][0])
            check("count_b", ft["count_b"][b], pt["count_b"][0])
            for li, (fs, ps) in enumerate(zip(ft["flow_steps"], pt["flow_steps"])):
                for k in ("h_in", "q", "k", "v", "msg", "out"):
                    check(f"step{li}.{k}", fs[k][:, b], ps[k][:, 0])
                for k in ("Q", "K", "V", "logits", "att", "key_mask"):
                    check(f"step{li}.{k}", fs[k][b], ps[k][0])
        print(f"\n[InfoFlow M1] max_err={max(errs.values()):.3e}")

    def test_backward_finite(self):
        d = self.delta.clone().requires_grad_(True)
        h, agree, cons = self.flow(d, self.mask)
        loss = h[self.mask].sum() + agree[self.mask].sum() + cons.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(d.grad[self.mask]).all())
        for p in self.flow.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_invalid_nan_backward_finite(self):
        d = self.delta.clone()
        d[~self.mask] = float("nan")
        d = d.requires_grad_(True)
        h, agree, cons = self.flow(d, self.mask)
        loss = h[self.mask].sum() + agree[self.mask].sum() + cons.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(d.grad[self.mask]).all())
        for p in self.flow.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == "__main__":
    unittest.main()
