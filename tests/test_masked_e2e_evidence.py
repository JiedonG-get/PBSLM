"""Independent CPU evidence for masked PBSLMv2; no training/optimizer steps.

M1 packs each sample's physical union and each role's support. Its comm,
Flow attention and statistics are explicit reference equations, not calls to
production masked comm/Flow. BlockBank and decoder layers are reused: this is
an execution/isolation reference, not an independent implementation of them.
Run: python -m unittest tests.test_masked_e2e_evidence -v
"""
import contextlib
import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.model_v2 import PBSLMv2

ATOL, RTOL = 1e-6, 1e-5


def scores(routes, k=8):
    out = torch.full((len(routes), k), -20.0)
    for b, route in enumerate(routes):
        for rank, physical_id in enumerate(route):
            out[b, physical_id] = 8.0 - rank
    return out


A_ROUTES = [[1, 5], [0, 7]]
C_ROUTES = [[3, 5], [2, 7]]


def model():
    torch.manual_seed(73)
    m = PBSLMv2(vocab=40, k=8, depth=1, d=16, heads=4,
                m_max=4, act_x=0.5, e_n=2, e_depth=1, refine=2,
                field_scale=0.0, use_mem=True, use_flow=True,
                use_sample_masks=True).cpu().eval()
    # Nonzero communications and nonuniform betas make wrong beta indexing observable.
    with torch.no_grad():
        m.bank.comm_beta.copy_(torch.linspace(-2, 2, 8))
        m.bank.comm2.weight.copy_(torch.linspace(-0.2, 0.3, 128).reshape(16, 8))
        m.bank.comm2.bias.copy_(torch.linspace(0.01, 0.08, 16))
        # Nonzero read-only fixture memory exercises A's E/M read without writes.
        for i, mem in enumerate(m.mem):
            mem.copy_(torch.eye(16) * (i + 1))
    return m


@contextlib.contextmanager
def fixed_scores(m, sa, sc):
    with patch.object(m.role_a, "forward", side_effect=lambda h: sa.to(h)), \
         patch.object(m.role_c, "forward", side_effect=lambda h: sc.to(h)):
        yield


def run_capture(m, ids, sa, sc, inject=None, gradients=False):
    """Capture real forward calls; wrappers preserve tensors and autograd graphs.

    inject=(stage, side, batch, physical_id) injects into a branch-local clone.
    Production Y/delta and the other role's input are never modified in place.
    """
    cap = {"comm": [], "flow": [], "side": [], "I": []}
    originals = (m.bank.block_forward, m.bank.comm, m.flow.forward,
                 m._side_cons_masked, m._side_I_masked, m.encode_masked)
    bf, comm, flow, side, side_i, encode = originals

    def block(x, idx):
        y = bf(x, idx)
        cap.update(idx=idx.clone(), x=x, Y=y, delta=y - x.unsqueeze(0))
        return y

    def contaminate(value, stage, which, mask):
        if inject is None or inject[:2] != (stage, which):
            return value
        _, _, b, physical_id = inject
        g = cap["idx"].tolist().index(physical_id)
        cap["injection_valid"] = bool(mask[g, b])
        value = value.clone()
        value[g, b] = float("nan")
        return value

    def wrapped_comm(y, g, idx=None, mask=None, trace=None):
        which = len(cap["comm"])
        t = {}
        y = contaminate(y, "comm", which, mask)
        out = comm(y, g, idx=idx, mask=mask, trace=t)
        cap["comm"].append(t)
        cap.setdefault("masks", []).append(mask.clone())
        return out

    def wrapped_flow(delta, mask=None, trace=None):
        which = len(cap["flow"])
        delta = contaminate(delta, "flow", which, mask)
        t = {}
        result = flow(delta, mask=mask, trace=t)
        cap["flow"].append(t)
        return result

    def wrapped_side(settled, score, chosen, rows):
        result = side(settled, score, chosen, rows)
        cap.setdefault("settled", []).append(settled)
        cap["side"].append(result)
        return result

    def wrapped_i(agree, rows):
        result = side_i(agree, rows)
        cap["I"].append(result)
        return result

    def wrapped_encode(*args, **kwargs):
        result = encode(*args, **kwargs)
        cap["enc"] = result
        return result

    with fixed_scores(m, sa, sc), \
         patch.object(m.bank, "block_forward", side_effect=block), \
         patch.object(m.bank, "comm", side_effect=wrapped_comm), \
         patch.object(m.flow, "forward", side_effect=wrapped_flow), \
         patch.object(m, "_side_cons_masked", side_effect=wrapped_side), \
         patch.object(m, "_side_I_masked", side_effect=wrapped_i), \
         patch.object(m, "encode_masked", side_effect=wrapped_encode), \
         (contextlib.nullcontext() if gradients else torch.no_grad()):
        cap["out"] = m(ids, 2, sample_noise=False)
    return cap


def reference_side(m, delta, physical_ids, score):
    """One sample, one role, packed valid physical blocks only."""
    count = len(physical_ids)
    w = torch.ones(count, 1, 1, 1)
    weighted = w * delta
    total = weighted.sum(0)
    others = total.unsqueeze(0) - weighted
    r = m.bank.comm2(F.gelu(m.bank.comm1(others)))
    beta = m.bank.comm_beta[physical_ids].sigmoid()[:, None, None, None]
    update = beta * w * r
    comm_out = delta + update
    ct = dict(valid_Y=delta, weighted_Y=weighted, tot=total, others=others,
              r=r, beta=beta, update=update, out=comm_out)
    h_proj = m.flow.proj_in(comm_out)
    h = h_proj
    steps = []
    for layer in m.flow.layers:
        # Deliberately use legacy mathematical einsum, independent of masked matmul code.
        q, k, v = layer.q(h), layer.k(h), layer.v(h)
        logits = torch.einsum("ibtd,jbtd->btij", q, k) / math.sqrt(q.shape[-1])
        att = logits.softmax(-1)
        msg = torch.einsum("btij,jbtd->ibtd", att, v)
        out = layer.ln(h + layer.o(msg))
        steps.append(dict(h_in=h, q=q, k=k, v=v, logits=logits,
                          att=att, msg=msg, out=out))
        h = out
    mean = h.sum(0) / count
    scale = h.norm(dim=-1).sum(0) / count + 1e-6
    agree = 1 - ((h - mean.unsqueeze(0)).norm(dim=-1) / scale.unsqueeze(0)).clamp(0, 2)
    cons = m.flow.proj_out(mean)
    settled = comm_out + m.flow.proj_out(h)
    selected_scores = score[physical_ids]
    weights = 0.9 / count + 0.1 * selected_scores.softmax(0)
    side = (settled * weights[:, None, None, None]).sum(0)
    ft = dict(proj_in=h_proj, h=h, mean=mean, scale=scale, agree=agree,
              cons=cons, flow_steps=steps)
    return dict(ids=physical_ids, comm=ct, flow=ft, settled=settled,
                side=side, I=agree.mean(0))


def reference_sample(m, ids, sa, sc):
    """M1 uses per-sample union (often shorter and differently indexed than M2)."""
    x = m.token_emb(ids)
    k_a, k_c = m.quotas()
    chosen = [sa.topk(k_a).indices.tolist(), sc.topk(k_c).indices.tolist()]
    union = sorted(set(chosen[0] + chosen[1]))
    y = m.bank.block_forward(x, torch.tensor(union))
    delta = y - x.unsqueeze(0)
    states = []
    for route, score in zip(chosen, (sa, sc)):
        # Pack side support in physical-ID order, not arbitrary topk order.
        route = sorted(route)
        rows = [union.index(j) for j in route]
        states.append(reference_side(m, delta[rows], route, score))
    _, memory = m.e_forward(x)
    enc_a = m.ln_a(x + states[0]["side"] + 0.1 * m.ea_proj(memory))
    enc_c = m.ln_c(x + states[1]["side"])
    outputs = {}
    for name, enc, query, pos, layers, refine, head, ctx in (
        ("a", enc_a, m.q_emb, m.q_pos, m.dec_a, m.refine_proj, m.out_a, m.ctx_a),
        ("c", enc_c, m.q_emb_c, m.q_pos_c, m.dec_c, m.refine_proj_c, m.out_c, m.ctx_c),
    ):
        q = (query[:2] + pos[:2]).unsqueeze(0)
        for iteration in range(m.refine):
            h = q
            for layer in layers:
                h = layer(h, enc)
            if iteration + 1 < m.refine:
                q = q + refine(h)
        x_ctx = x[:, -1] + x.mean(1)
        outputs["z_" + name] = F.normalize(head(h) + ctx(x_ctx).unsqueeze(1), dim=-1)
    return dict(union=union, Y=y, delta=delta, states=states,
                a_enc=enc_a, c_enc=enc_c, **outputs)


class MaskedE2EEvidence(unittest.TestCase):
    def setUp(self):
        self.m = model()
        self.ids = torch.tensor([[1, 2, 3, 4, 5], [9, 10, 11, 12, 13]])
        self.sa, self.sc = scores(A_ROUTES), scores(C_ROUTES)
        self.errors = {}
        self.initial_state = {k: v.detach().clone() for k, v in self.m.state_dict().items()}

    def tearDown(self):
        for key, value in self.m.state_dict().items():
            self.assertTrue(torch.equal(value, self.initial_state[key]), key + " was updated")
        self.assertTrue(all(p.grad is None for p in self.m.parameters()))

    def close(self, label, a, b):
        self.assertEqual(tuple(a.shape), tuple(b.shape), label)
        torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL, msg=label)
        if a.numel():
            self.assertTrue(torch.isfinite(a).all(), label)
            self.errors[label] = max(self.errors.get(label, 0.0), (a - b).abs().max().item())

    def compare_side(self, cap, ref, b, side):
        global_ids = cap["idx"].tolist()
        rows = [global_ids.index(j) for j in ref["ids"]]
        ct, ft = cap["comm"][side], cap["flow"][side]
        for key in ("valid_Y", "weighted_Y", "others", "r", "update", "out"):
            self.close(f"s{side}.comm.{key}", ct[key][rows, b:b+1], ref["comm"][key])
        self.close("beta", ct["beta"][rows], ref["comm"]["beta"])
        self.close("tot", ct["tot"][b:b+1], ref["comm"]["tot"])
        for key in ("proj_in", "h", "agree"):
            self.close(f"flow.{key}", ft[key][rows, b:b+1], ref["flow"][key])
        for key in ("mean", "scale", "cons"):
            self.close(key, ft[key][b:b+1], ref["flow"][key])
        for n, (actual, expected) in enumerate(zip(ft["flow_steps"], ref["flow"]["flow_steps"])):
            for key in ("h_in", "q", "k", "v", "msg", "out"):
                self.close(f"step{n}.{key}", actual[key][rows, b:b+1], expected[key])
            # Compare key validity/-inf separately; never subtract -inf from -inf.
            valid = cap["masks"][side][:, b]
            self.assertTrue(torch.isneginf(actual["logits"][b, ..., ~valid]).all())
            self.assertTrue((actual["att"][b, ..., ~valid] == 0).all())
            for key in ("logits", "att"):
                square = actual[key][b:b+1].index_select(2, torch.tensor(rows)).index_select(3, torch.tensor(rows))
                self.close(f"step{n}.{key}", square, expected[key])
            self.assertTrue((actual["out"][~cap["masks"][side]] == 0).all())
        self.close("settled", cap["settled"][side][rows, b:b+1], ref["settled"])
        self.close("side", cap["side"][side][b:b+1], ref["side"])
        self.close("I", cap["I"][side][b:b+1], ref["I"])

    def test_packed_physical_id_m1_full_chain(self):
        cap = run_capture(self.m, self.ids, self.sa, self.sc)
        with torch.no_grad():
            for b in range(2):
                ref = reference_sample(self.m, self.ids[b:b+1], self.sa[b], self.sc[b])
                self.assertLess(len(ref["union"]), len(cap["idx"]))
                rows = [cap["idx"].tolist().index(j) for j in ref["union"]]
                self.close("Y", cap["Y"][rows, b:b+1], ref["Y"])
                self.close("delta", cap["delta"][rows, b:b+1], ref["delta"])
                for side in range(2):
                    self.compare_side(cap, ref["states"][side], b, side)
                for key in ("a_enc", "c_enc"):
                    self.close(key, cap["enc"][key][b:b+1], ref[key])
                for key in ("z_a", "z_c"):
                    self.close(key, cap["out"][key][b:b+1], ref[key])
        print("\n[M1 physical-ID aligned max_abs]", self.errors)

    def fixed_side_equal(self, base, altered, side, b):
        name = "a" if side == 0 else "c"
        self.assertEqual(base["out"]["chosen_" + name][b], altered["out"]["chosen_" + name][b])
        phys = sorted(base["out"]["chosen_" + name][b])
        left = [base["idx"].tolist().index(j) for j in phys]
        right = [altered["idx"].tolist().index(j) for j in phys]
        for group in ("comm", "flow"):
            keys = ("others", "r", "update", "out") if group == "comm" else ("proj_in", "h", "agree")
            for key in keys:
                self.close(group + key, base[group][side][key][left, b], altered[group][side][key][right, b])
        for a, c in zip(base["flow"][side]["flow_steps"], altered["flow"][side]["flow_steps"]):
            for key in ("q", "k", "v", "msg", "out"):
                self.close(key, a[key][left, b], c[key][right, b])
            for key in ("logits", "att"):
                lhs = a[key][b].index_select(1, torch.tensor(left)).index_select(2, torch.tensor(left))
                rhs = c[key][b].index_select(1, torch.tensor(right)).index_select(2, torch.tensor(right))
                self.close(key, lhs, rhs)
        for key in ("mean", "scale", "cons"):
            self.close(key, base["flow"][side][key][b], altered["flow"][side][key][b])
        for key in ("settled", "side", "I"):
            x, y = base[key][side], altered[key][side]
            self.close(key, x[left, b] if key == "settled" else x[b],
                       y[right, b] if key == "settled" else y[b])
        self.close(name + "_enc", base["enc"][name + "_enc"][b], altered["enc"][name + "_enc"][b])
        self.close("z_" + name, base["out"]["z_" + name][b], altered["out"]["z_" + name][b])

    def test_bilateral_route_changes_and_other_sample(self):
        base = run_capture(self.m, self.ids, self.sa, self.sc)
        for changed in (0, 1):
            sa, sc = self.sa.clone(), self.sc.clone()
            (sa if changed == 0 else sc)[0] = scores([[4, 6]])[0]
            other = run_capture(self.m, self.ids, sa, sc)
            key = "chosen_a" if changed == 0 else "chosen_c"
            self.assertNotEqual(base["out"][key][0], other["out"][key][0])
            self.fixed_side_equal(base, other, 1 - changed, 0)
            for side in (0, 1):
                self.fixed_side_equal(base, other, side, 1)

    def test_overlap_partial_disjoint_in_actual_forward(self):
        for a, c in (([1, 5], [1, 5]), ([1, 5], [3, 5]), ([1, 5], [3, 6])):
            with self.subTest(a=a, c=c):
                cap = run_capture(self.m, self.ids, scores([a, [0, 7]]), scores([c, [2, 7]]))
                self.assertEqual(cap["out"]["chosen_a"][0], a)
                self.assertEqual(cap["out"]["chosen_c"][0], c)
                union = sorted(set(a + c + [0, 7, 2, 7]))
                self.assertEqual(cap["idx"].tolist(), union)
                for side, route in enumerate((a, c)):
                    actual = {j for g, j in enumerate(union) if cap["masks"][side][g, 0]}
                    self.assertEqual(actual, set(route))
                self.assertNotEqual(cap["comm"][0]["out"].data_ptr(), cap["comm"][1]["out"].data_ptr())
                self.assertNotEqual(cap["flow"][0]["h"].data_ptr(), cap["flow"][1]["h"].data_ptr())

    def probes(self, out):
        # Nonuniform fixed probes avoid degeneracy of summing normalized vectors.
        weights = torch.linspace(-0.73, 1.17, out["z_a"].numel()).reshape_as(out["z_a"])
        return [(out["z_a"] * weights).sum(), (out["z_c"] * weights.flip(-1)).sum()]

    def test_side_specific_invalid_nan_forward_and_autograd(self):
        baseline = run_capture(self.m, self.ids, self.sa, self.sc)
        # sample0: block3 A-invalid/C-valid; block1 C-invalid/A-valid;
        # block0 selected only by sample1 => invalid for both roles in sample0.
        for stage in ("comm", "flow"):
            for side, physical in ((0, 3), (1, 1), (0, 0), (1, 0)):
                with self.subTest(stage=stage, side=side, physical=physical):
                    cap = run_capture(self.m, self.ids, self.sa, self.sc,
                                      inject=(stage, side, 0, physical), gradients=True)
                    self.assertFalse(cap["injection_valid"])
                    for fixed_side in (0, 1):
                        for b in (0, 1):
                            self.fixed_side_equal(baseline, cap, fixed_side, b)
                    params = [p for p in self.m.parameters() if p.requires_grad]
                    for probe in self.probes(cap["out"]):
                        grads = torch.autograd.grad(probe, [cap["x"], *params], retain_graph=True,
                                                    allow_unused=True)
                        self.assertIsNotNone(grads[0])
                        self.assertTrue(torch.isfinite(grads[0]).all())
                        self.assertTrue(any(g is not None and g.abs().max() > 1e-8 for g in grads))
                        for grad in grads:
                            if grad is not None:
                                self.assertTrue(torch.isfinite(grad).all())

    def test_side_valid_nan_reports_boundary(self):
        for stage, boundary in (("comm", "comm.valid_Y"), ("flow", "InfoFlow.delta_in")):
            for side, physical in ((0, 1), (1, 3)):
                with self.subTest(stage=stage, side=side):
                    with self.assertRaisesRegex(FloatingPointError, boundary):
                        run_capture(self.m, self.ids, self.sa, self.sc, inject=(stage, side, 0, physical))

    def test_nonuniform_probes_gradient_classification(self):
        # Natural routers here: unlike fixed-score tests this preserves router gradients.
        with torch.enable_grad():
            out = self.m(self.ids, 2, sample_noise=False)
            probes = self.probes(out)
            named = [(n, p) for n, p in self.m.named_parameters() if p.requires_grad]
            values = [torch.autograd.grad(q, [p for _, p in named], retain_graph=True,
                                          allow_unused=True) for q in probes]
        records = {}
        for (name, _), ga, gc in zip(named, *values):
            entry = []
            for grad in (ga, gc):
                entry.append(dict(is_none=grad is None,
                                  isfinite=None if grad is None else bool(torch.isfinite(grad).all()),
                                  max_abs_grad=None if grad is None else float(grad.abs().max())))
                if grad is not None:
                    self.assertTrue(torch.isfinite(grad).all(), name)
            records[name] = entry
        for shared in ("tok.weight", "bank.qkv_w", "bank.comm2.weight", "flow.proj_in.weight"):
            self.assertTrue(all(not e["is_none"] and e["max_abs_grad"] > 1e-8 for e in records[shared]), shared)
        for name in ("out_a.weight", "ctx_a.weight", "dec_a.0.q.weight"):
            self.assertIsNotNone(records[name][0]["max_abs_grad"])
            self.assertTrue(records[name][1]["is_none"], name)
        for name in ("out_c.weight", "ctx_c.weight", "dec_c.0.q.weight"):
            self.assertTrue(records[name][0]["is_none"], name)
            self.assertIsNotNone(records[name][1]["max_abs_grad"])
        # E/M belongs to A's read branch; C does not use it.
        self.assertTrue(records["ea_proj.weight"][1]["is_none"])
        self.assertGreater(records["ea_proj.weight"][0]["max_abs_grad"], 1e-8)
        print("\n[gradient classification A,C]", records)
        self.assertTrue(all(p.grad is None for p in self.m.parameters()))


if __name__ == "__main__":
    unittest.main()
