"""v2 语言模型：自训 BPE 16k + 连续向量输出（无 softmax）+ 可学习扰动场（三旋钮）。

输出不是 V 词表概率分布，而是每个位置一个 d 维连续向量 z：
  z = Output(Flow/A) + Field(activity, shock, emotion)
训练目标：MSE(z, E[y])；解码时最近邻到 token 嵌入 E。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BlockBank, CrossLayer, InfoFlow

EMO_CALM = [0.55, 0.15, 0.05, 0.55, 0.0, 0.0, 0.0, 0.0]


class PerturbField(nn.Module):
    """三个旋钮：activity、shock、emotion(8) -> 高斯 μ/σ（d 维）。"""

    def __init__(self, d, emo_dim=8, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + emo_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * d),
        )

    def forward(self, act, shock, emo):
        h = torch.cat([act, shock, emo], dim=-1)
        out = self.net(h)
        mu, log_sigma = out[..., :self.net[-1].out_features // 2], \
            out[..., self.net[-1].out_features // 2:]
        return mu, log_sigma.clamp(-6, 2)


class PBSLMv2(nn.Module):
    def __init__(self, vocab, k=30, depth=4, d=768, heads=12, m_max=128,
                 act_x=0.2, e_n=10, e_depth=1, emo_dim=8, refine=2, field_scale=0.0,
                 use_mem=True, trainable_target=True, simple_dec=False,
                 dense=False, use_flow=True, direct_only=False):
        super().__init__()
        self.vocab, self.k, self.depth, self.d, self.heads = vocab, k, depth, d, heads
        self.act_x, self.e_n, self.emo_dim = act_x, e_n, emo_dim
        self.field_scale = field_scale
        self.use_mem = use_mem
        self.trainable_target = trainable_target
        self.simple_dec = simple_dec
        self.dense = dense
        self.use_flow = use_flow
        self.direct_only = direct_only
        self.refine = refine
        # 输入 token 嵌入（可训练、单位归一化）
        self.tok = nn.Embedding(vocab, d)
        # 输出码本（冻结随机单位向量，防止模型通过移动目标嵌入作弊）
        code = F.normalize(torch.randn(vocab, d), dim=-1)
        self.register_buffer("code", code)   # 必须持久化：否则加载 ckpt 会得到一套新的随机码本，解码全错
        self.bank = BlockBank(k, depth, d, heads)
        self.ebank = BlockBank(e_n, e_depth, d, heads)
        self.role_a = nn.Linear(d, k, bias=False)
        self.role_c = nn.Linear(d, k, bias=False)
        self.flow = InfoFlow(d, 64, 2)
        self.e_mix = nn.Parameter(torch.zeros(e_n, e_n))
        self.ea_proj = nn.Linear(d, d)
        nn.init.normal_(self.ea_proj.weight, std=0.02)
        nn.init.zeros_(self.ea_proj.bias)
        self.mem = nn.ParameterList([nn.Parameter(torch.zeros(d, d), requires_grad=False)
                                     for _ in range(e_n)])
        self.q_emb = nn.Parameter(torch.randn(m_max, d) * 0.02)
        self.q_pos = nn.Parameter(torch.randn(m_max, d) * 0.02)
        self.q_emb_c = nn.Parameter(torch.randn(m_max, d) * 0.02)
        self.q_pos_c = nn.Parameter(torch.randn(m_max, d) * 0.02)
        # v2 的 query 是固定 q0，cross-attn 的上下文信号容易被残差淹没 → 放大 cross-attn 残差
        self.dec_a = nn.ModuleList([CrossLayer(d, heads, True, res_scale=4.0) for _ in range(2)])
        self.dec_c = nn.ModuleList([CrossLayer(d, heads, True, res_scale=4.0) for _ in range(1)])
        self.refine_proj = nn.Linear(d, d)
        self.refine_proj_c = nn.Linear(d, d)
        self.out_a = nn.Linear(d, d)
        nn.init.eye_(self.out_a.weight)
        # 直接上下文读出（给编码器/主块一条短梯度路径，避免 decoder 残差把上下文信号淹没）
        self.ctx_a = nn.Linear(d, d)
        self.ctx_c = nn.Linear(d, d)
        nn.init.zeros_(self.out_a.bias)
        self.a_pred = nn.Linear(d, d)
        self.ln_a, self.ln_c = nn.LayerNorm(d), nn.LayerNorm(d)
        self.field = PerturbField(d, emo_dim)
        self.register_buffer("emo_calm", torch.tensor(EMO_CALM))

    # ---- 与 v15 相同的编码（A/C 角色互斥、E 常驻、M 只进 A）
    def quotas(self, act=None):
        act = self.act_x if act is None else act
        cap = max(3, int(round(self.act_x * self.k)))
        m_t = max(3, min(int(round(float(act) * self.k)), cap, self.k))
        kA = max(1, m_t // 2)
        return kA, max(1, m_t - kA)

    def _select(self, score_a, score_c, kA, kC):
        taken = torch.zeros(self.k, dtype=torch.bool, device=score_a.device)
        ia = torch.topk(score_a, kA)[1]
        taken = taken.clone(); taken[ia] = True
        ic = torch.topk(score_c.masked_fill(taken, -torch.finfo(score_c.dtype).max), kC)[1]
        return ia.tolist(), ic.tolist(), score_a, score_c

    def e_forward(self, x):
        E_out = self.ebank.block_forward(x, torch.arange(self.e_n, device=x.device))
        E_out = self.ebank.comm(E_out, torch.ones(self.e_n, device=x.device, dtype=x.dtype))
        mix = torch.einsum("ebtd,ef->fbtd", E_out, self.e_mix)
        E_mix = E_out + mix
        # 记忆矩阵范数会随写入增长；读之前先归一化，避免 A 编码被一个巨大的常量记忆向量吞掉
        mem_n = [F.normalize(self.mem[i], dim=-1) for i in range(self.e_n)]
        reads = torch.stack([torch.einsum("btd,dv->btv", E_mix[i], mem_n[i])
                             for i in range(self.e_n)], dim=0)
        return E_mix, reads.mean(0)

    def token_emb(self, ids):
        return F.normalize(self.tok.weight[ids.long()], dim=-1)

    def target_embed(self, ids):
        if self.trainable_target:
            return F.normalize(self.tok.weight[ids.long()], dim=-1)
        return self.code[ids.long()]

    def encode(self, ids, act=None):
        x = self.token_emb(ids)
        kA, kC = self.quotas(act)
        hmean = x.mean(1).mean(0)
        sa, sc = self.role_a(hmean), self.role_c(hmean)
        if self.dense:
            allb = list(range(self.k))
            ia, ic = allb, allb
        else:
            ia, ic, sa, sc = self._select(sa, sc, kA, kC)
        chosen = [ia, ic]
        active = sorted(set(ia + ic))
        Y = self.bank.block_forward(x, torch.tensor(active, device=x.device))
        delta = Y - x.unsqueeze(0)
        delta = self.bank.comm(delta, torch.ones(len(active), device=x.device, dtype=x.dtype))
        if self.use_flow:
            settled_low, agree, _ = self.flow(delta)
            settled = delta + self.flow.proj_out(settled_low)  # flow 残差，投影回 d
        else:
            settled = delta
            agree = torch.ones(len(active), x.shape[0], x.shape[1],
                               device=x.device, dtype=x.dtype)
        row = {j: i for i, j in enumerate(active)}

        def side_cons(which, score):
            rows = [row[j] for j in chosen[which]]
            kk = len(rows)
            w = 0.9 * torch.full((kk,), 1.0 / kk, device=x.device, dtype=score.dtype) \
                + 0.1 * torch.softmax(score[chosen[which]], dim=0)
            return (settled[rows] * w[:, None, None, None]).sum(0)

        def side_I(which):
            rows = [row[j] for j in chosen[which]]
            return agree[rows].mean(0)

        E_feats, mem_read = self.e_forward(x)
        if self.use_mem:
            a_enc = self.ln_a(x + side_cons(0, sa) + 0.1 * self.ea_proj(mem_read))
        else:
            a_enc = self.ln_a(x + side_cons(0, sa))
        c_enc = self.ln_c(x + side_cons(1, sc))
        return a_enc, c_enc, E_feats, side_I(0), side_I(1), active

    def _decode(self, layers, q, kv):
        h = q
        for layer in layers:
            h = layer(h, kv)
        return h

    def _decode_refine(self, layers, q0, kv, proj):
        qq = q0
        for r in range(self.refine):
            h = self._decode(layers, qq, kv)
            if r + 1 < self.refine:
                qq = qq + proj(h)
        return h

    def forward(self, ids, m, act=None, emo=None, shock=None, sample_noise=True):
        B = ids.shape[0]
        emo = (self.emo_calm.expand(B, -1) if emo is None else emo).to(self.tok.weight.dtype)
        a_enc, c_enc, E_feats, I_a, I_c, active = self.encode(ids, act)
        q0 = (self.q_emb[:m] + self.q_pos[:m]).unsqueeze(0).expand(B, m, self.d)
        q0c = (self.q_emb_c[:m] + self.q_pos_c[:m]).unsqueeze(0).expand(B, m, self.d)
        if self.simple_dec:
            h_a = (a_enc[:, -1] + a_enc.mean(1)).unsqueeze(1) + self.q_pos[:m].unsqueeze(0)
            h_c = (c_enc[:, -1] + c_enc.mean(1)).unsqueeze(1) + self.q_pos_c[:m].unsqueeze(0)
        else:
            h_a = self._decode_refine(self.dec_a, q0, a_enc, self.refine_proj)
            h_c = self._decode_refine(self.dec_c, q0c, c_enc, self.refine_proj_c)
        # 输出必须先归一化：否则 cos 损失下 z 范数会爆（162+），梯度 1/||z|| 衰减→学习停滞
        x_all = self.token_emb(ids)
        x_ctx = x_all[:, -1] + x_all.mean(1)   # 同时用最后 token 与全文上下文，避免退化成 bigram
        ctx_a = self.ctx_a(x_ctx).unsqueeze(1)
        ctx_c = self.ctx_c(x_ctx).unsqueeze(1)
        if self.direct_only:
            z_a = F.normalize(ctx_a.expand(-1, m, -1), dim=-1)
            z_c = F.normalize(ctx_c.expand(-1, m, -1), dim=-1)
        else:
            z_a = F.normalize(self.out_a(h_a) + ctx_a, dim=-1)
            z_c = F.normalize(self.out_a(h_c) + ctx_c, dim=-1)
        # 三旋钮扰动场：activity / shock / emotion；shock 由训练脚本算好后传入（B,1）
        dt = self.tok.weight.dtype
        act_b = torch.full((B, 1), act / self.act_x if act else 1.0, device=ids.device, dtype=dt)
        shock_b = torch.zeros(B, 1, device=ids.device, dtype=dt) if shock is None else shock.to(dt)
        mu_raw, log_sigma = self.field(act_b, shock_b, emo)
        # 扰动场只提供“方向 + 幅度”，不能让它吞掉输出：μ 单位化、整体缩放固定很小
        mu = F.normalize(mu_raw, dim=-1)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu) if sample_noise else torch.zeros_like(mu)
        if self.field_scale > 0:
            z_pert = F.normalize(
                z_a + self.field_scale * (mu.unsqueeze(1) + (sigma * eps).unsqueeze(1)),
                dim=-1)
        else:
            z_pert = z_a
        return {"z_a": z_a, "z_c": z_c, "field_mu": mu, "field_log_sigma": log_sigma,
                "z_pert": z_pert,
                "h_a": h_a, "e_feats": E_feats, "I_a": I_a, "I_c": I_c, "active": active}

    @torch.no_grad()
    def write_memory(self, e_feats, values, gate=1.0, lr=0.05):
        k = F.normalize(e_feats.mean(2), dim=-1)
        v = F.normalize(values.mean(1), dim=-1)
        for i in range(self.e_n):
            self.mem[i].add_(gate * lr * torch.einsum("bd,bv->dv", k[i], v))

    @torch.no_grad()
    def decay_memory(self, decay=0.98):
        for i in range(self.e_n):
            self.mem[i].mul_(decay)

    @torch.no_grad()
    def memory_norms(self):
        return [float(m.norm()) for m in self.mem]

    @torch.no_grad()
    def decode_nearest(self, z):
        zz = F.normalize(z.float(), dim=-1)
        E = F.normalize(self.tok.weight.float(), dim=-1) if self.trainable_target else self.code.float()
        return torch.matmul(zz, E.t())
