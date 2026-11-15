"""O1 测试：输出拆分、在线白名单、C 稳定性、role_a 耦合。

只做 forward / 参数赋值 / param_groups 检查；不计算 loss、不 backward、不 optimizer.step。
"""
import copy
import hashlib
import unittest

import torch
import torch.nn.functional as F

from models.model_v2 import PBSLMv2


def make_model(direct_only=False, field_scale=0.0, seed=0):
    torch.manual_seed(seed)
    m = PBSLMv2(vocab=64, k=8, depth=1, d=32, heads=4, m_max=8,
                act_x=0.2, e_n=2, e_depth=1, emo_dim=8, field_scale=field_scale,
                use_mem=True, trainable_target=True, simple_dec=False,
                dense=False, use_flow=True, direct_only=direct_only)
    m.eval()
    return m


def forward(m, ids, act=0.2):
    with torch.no_grad():
        return m(ids, 2, act=act, sample_noise=False)


def c_topk(m, out, k=5):
    """C 输出的最近邻 top-k（完整 batch × m；只用于评估，不产生训练信号）。"""
    E = F.normalize(m.code.float(), dim=-1)
    z = F.normalize(out["z_c"].reshape(-1, out["z_c"].shape[-1]).float(), dim=-1)
    return (z @ E.t()).topk(k, dim=-1).indices   # (B*m, k)


ALLOWED_PREFIX = ("dec_a.", "refine_proj.", "ln_a.", "ctx_a.", "out_a.")
ALLOWED_EXACT = {"q_emb", "q_pos"}


def is_allowed_param(name):
    return name in ALLOWED_EXACT or any(name.startswith(p) for p in ALLOWED_PREFIX)


def protect_hash(m):
    h = hashlib.sha256()
    for name, v in sorted(m.state_dict().items()):
        if is_allowed_param(name):
            continue
        h.update(name.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class O1Tests(unittest.TestCase):
    def setUp(self):
        self.ids = torch.arange(32).reshape(2, 16) % 64
        self.model = make_model()
        self.base = forward(self.model, self.ids)

    def test_out_layers_init_and_direct_only(self):
        # out_c 初始化必须复制 out_a 的 weight/bias
        self.assertTrue(torch.equal(self.model.out_c.weight, self.model.out_a.weight))
        self.assertTrue(torch.equal(self.model.out_c.bias, self.model.out_a.bias))
        # direct_only=True 路径也必须能前向
        m2 = make_model(direct_only=True)
        o = forward(m2, self.ids)
        self.assertEqual(o["z_a"].shape[0], 2)
        self.assertEqual(o["z_c"].shape[0], 2)

    def test_forward_baseline(self):
        o = self.base
        self.assertIn("chosen", o)
        self.assertEqual(len(o["chosen"]), 2)       # A, C
        self.assertEqual(len(o["active"]), 3)       # act=0.2 -> cap 3
        self.assertEqual(o["z_c"].shape, (2, 2, 32))
        self.assertFalse(torch.isnan(o["z_c"]).any())
        self.assertFalse(self.model.direct_only)     # 默认覆盖完整 out_a/out_c 路径
        self.assertEqual(c_topk(self.model, o).shape, (4, 5))

    def test_param_groups_whitelist(self):
        m = make_model()
        groups = m.online_param_groups()
        expected = {"dec_a", "a_query", "refine_proj", "ln_a", "ctx_a", "out_a"}
        self.assertEqual(set(groups), expected)
        allowed = m.online_parameters()
        self.assertTrue(all(p.requires_grad for p in allowed))
        # 冻结：role_a、C、out_c、shared 不得进入白名单
        frozen = [m.role_a.weight, m.role_c.weight, m.out_c.weight,
                  m.tok.weight, m.bank.qkv_w, m.flow.proj_in.weight,
                  m.field.net[0].weight, m.a_pred.weight]
        allowed_ids = {id(p) for p in allowed}
        self.assertFalse(any(id(p) in allowed_ids for p in frozen))
        m.freeze_for_online()
        opt = torch.optim.AdamW(m.online_parameters(), lr=1e-3)  # 只构造，不 step
        self.assertEqual(len(opt.param_groups), 1)
        opt_ids = {id(p) for p in opt.param_groups[0]["params"]}
        self.assertEqual(opt_ids, allowed_ids)
        self.assertNotIn(id(m.role_a.weight), opt_ids)

    def _assert_c_stable_after_groups(self, m, seed=123):
        """field 任意开关下都逐组扰动六组 A 参数，校验 C 的全部指标。"""
        m.freeze_for_online()
        base = forward(m, self.ids)
        base_state = copy.deepcopy(m.state_dict())
        base_protected = protect_hash(m)
        base_ck = (list(base["chosen"][1]), list(base["active"]),
                   base["z_c"].clone(), base["I_c"].clone(),
                   c_topk(m, base).clone())
        g = torch.Generator().manual_seed(seed)
        for gname, params in m.online_param_groups().items():
            with torch.no_grad():
                for p in params:
                    p.add_(torch.randn_like(p, generator=g) * 0.01)
            out = forward(m, self.ids)
            self.assertEqual(out["chosen"][1], base_ck[0], f"C chosen changed: {gname}")
            self.assertEqual(out["active"], base_ck[1], f"active changed: {gname}")
            self.assertTrue(torch.allclose(out["z_c"], base_ck[2], atol=1e-6, rtol=1e-5),
                            f"z_c changed: {gname}")
            self.assertTrue(torch.allclose(out["I_c"], base_ck[3], atol=1e-6, rtol=1e-5),
                            f"I_c changed: {gname}")
            self.assertTrue(torch.equal(c_topk(m, out), base_ck[4]),
                            f"C top-k changed: {gname}")
            self.assertEqual(protect_hash(m), base_protected, f"protected hash changed: {gname}")
            m.load_state_dict(base_state)

    def test_c_stability_for_allowed_groups(self):
        self._assert_c_stable_after_groups(make_model(field_scale=0.0), seed=123)

    def test_role_a_coupling_diagnostic(self):
        m = make_model()
        m.freeze_for_online()
        base = forward(m, self.ids)
        base_state = copy.deepcopy(m.state_dict())
        base_ck = (list(base["chosen"][1]), list(base["active"]),
                   base["z_c"].clone(), base["I_c"].clone(),
                   c_topk(m, base).clone())
        g = torch.Generator().manual_seed(321)
        a_changed = False
        c_changed = False
        for scale in (0.1, 0.3, 1.0, 3.0, 10.0):
            m.load_state_dict(base_state)
            with torch.no_grad():
                m.role_a.weight.add_(torch.randn_like(m.role_a.weight, generator=g) * scale)
            out = forward(m, self.ids)
            if sorted(out["chosen"][0]) != sorted(base["chosen"][0]):
                a_changed = True
                c_changed = (
                    list(out["chosen"][1]) != base_ck[0]
                    or list(out["active"]) != base_ck[1]
                    or not torch.allclose(out["z_c"], base_ck[2], atol=1e-6, rtol=1e-5)
                    or not torch.allclose(out["I_c"], base_ck[3], atol=1e-6, rtol=1e-5)
                    or not torch.equal(c_topk(m, out), base_ck[4])
                )
                break
        self.assertTrue(a_changed, "role_a perturbation did not change A selection; test setup invalid")
        # 不断言 C 必然变化；打印诊断，供人工分析
        print(f"\n[role_a] role_a_setup_ok={a_changed} A_selection_changed={a_changed} "
              f"C_changes={c_changed}")
        print(f"  base A={base['chosen'][0]} C={base['chosen'][1]} active={base['active']}")

    def test_m_written_does_not_alter_c(self):
        m = make_model()
        m.freeze_for_online()
        base = forward(m, self.ids)
        base_ck = (list(base["chosen"][1]), list(base["active"]),
                   base["z_c"].clone(), base["I_c"].clone(),
                   c_topk(m, base).clone())
        values = m.code[self.ids]          # 任意确定性 value；只用于测试 C 不读 M
        m.write_memory(base["e_feats"], values, gate=1.0, lr=0.05)
        out = forward(m, self.ids)
        self.assertEqual(list(out["chosen"][1]), base_ck[0])
        self.assertEqual(list(out["active"]), base_ck[1])
        self.assertTrue(torch.allclose(out["z_c"], base_ck[2], atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(out["I_c"], base_ck[3], atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.equal(c_topk(m, out), base_ck[4]))

    def test_field_on_c_stability(self):
        # field_scale=0.1 时也必须覆盖全部六组，而不是只扰动 ctx_a
        self._assert_c_stable_after_groups(make_model(field_scale=0.1), seed=456)

    def test_load_state_dict_compat(self):
        m = make_model()
        state = copy.deepcopy(m.state_dict())
        old = {k: v for k, v in state.items() if not k.startswith("out_c.")}
        m2 = make_model()
        m2.load_state_dict(old)
        self.assertTrue(torch.equal(m2.out_c.weight, old["out_a.weight"]))
        self.assertTrue(torch.equal(m2.out_c.bias, old["out_a.bias"]))

    def test_direct_only_c_stability(self):
        m = make_model(direct_only=True)
        m.freeze_for_online()
        base = forward(m, self.ids)
        base_z = base["z_c"].clone()
        base_ck = list(base["chosen"][1])
        base_topk = c_topk(m, base).clone()
        base_state = copy.deepcopy(m.state_dict())
        with torch.no_grad():
            m.ctx_a.weight.add_(torch.randn_like(m.ctx_a.weight) * 0.01)
        out = forward(m, self.ids)
        self.assertEqual(list(out["chosen"][1]), base_ck)
        self.assertTrue(torch.allclose(out["z_c"], base_z, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.equal(c_topk(m, out), base_topk))
        m.load_state_dict(base_state)


if __name__ == "__main__":
    unittest.main()
