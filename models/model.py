"""v13：正式在线模型（信息流动一致性 + 震撼值 + 脑活跃度调度）。

在 v12（PBS-LM：可调深度 + n→m）基础上，按用户 11-13 晚的修订实现三件事：

1. **信息一致性 = 流动模型**（替换 pairwise TV）：
   活跃块的信息投影到低维流动空间，做 T 步可学习消息传递（动力学），
   得到 settled 场；一致性 = 各块 settled 状态离共识场的归一化离散度（越小越一致）。
   A/C 各自的一致性 = 各自角色块相对全局 settled 场的平均一致度，比较后仲裁。

2. **震撼值（冲击评分）**：
   S_t = -log(known + eps)，known = 1 - (1-p_A(y))(1-p_C(y))
   —— 正确答案任意一侧"本来就知道/想到了" -> known 高 -> 震撼低；两侧都不知道 -> 震撼高。
   训练/在线损失用它当乘数（权重 = 1 + β·S），替代原来的 CE 比例占位。

3. **脑活跃度**：
   A_t = act_x · σ(a(S−s0) + b(1−I) − c)，默认 act_x=0.2（最多 20% 块启用）。
   低震撼 + 高一致 -> 低活跃；否则高活跃。活跃度 -> 启用块数 m_t=clamp(round(A·K),3,⌊act_x·K⌋)
   -> k1/k2/k3 配额。未启用的块完全不参与计算（省延迟）。

另外：信息一致性去掉了原来的"额外解码块视图"（省两次解码）；块间交流残差保留。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PHI = (1.0 + 5.0 ** 0.5) / 2.0


# ---------------------------------------------------------------- 基础件
def _ln(x, g, b):
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + 1e-5) * g[:, None, None, :] + b[:, None, None, :]


def hrr_bind(a, b):
    d = a.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(a, dim=-1) * torch.fft.rfft(b, dim=-1), n=d, dim=-1)


def hrr_unbind(m, q):
    d = m.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(m, dim=-1)[:, None, :]
                           * torch.conj(torch.fft.rfft(q, dim=-1)), n=d, dim=-1)


def _check_finite(t, label):
    if not torch.isfinite(t).all():
        raise FloatingPointError(f"non-finite value at {label}")


def _check_valid_finite(t, mask, label):
    """Only valid blocks/positions may contain NaN/Inf; invalid ones must be cleaned first."""
    if t.numel() == 0:
        return
    if mask.dim() == 2:
        vals = t[mask]
    else:
        vals = t
    if not torch.isfinite(vals).all():
        raise FloatingPointError(f"non-finite value in valid region at {label}")


class BlockBank(nn.Module):
    """nb 个权重独立的真块；每个块 = depth 个子层（子层各自独立权重）。
    forward 只计算传入 idx 的块（活跃块），按块批量 bmm。"""

    def __init__(self, nb, depth=1, d=256, heads=8, pe_k=12):
        super().__init__()
        self.nb, self.depth, self.d, self.heads = nb, depth, d, heads
        self.hd = d // heads
        nb_, dp_, d_ = nb, depth, d
        self.qkv_w = nn.Parameter(torch.randn(nb_, dp_, d, 3 * d) * d ** -0.5)
        self.qkv_b = nn.Parameter(torch.zeros(nb_, dp_, 3 * d))
        self.out_w = nn.Parameter(torch.randn(nb_, dp_, d, d) * d ** -0.5)
        self.out_b = nn.Parameter(torch.zeros(nb_, dp_, d))
        self.ff1_w = nn.Parameter(torch.randn(nb_, dp_, d, 4 * d) * d ** -0.5)
        self.ff1_b = nn.Parameter(torch.zeros(nb_, dp_, 4 * d))
        self.ff2_w = nn.Parameter(torch.randn(nb_, dp_, 4 * d, d) * (4 * d) ** -0.5)
        self.ff2_b = nn.Parameter(torch.zeros(nb_, dp_, d))
        self.ln1_g = nn.Parameter(torch.ones(nb_, dp_, d))
        self.ln1_b = nn.Parameter(torch.zeros(nb_, dp_, d))
        self.ln2_g = nn.Parameter(torch.ones(nb_, dp_, d))
        self.ln2_b = nn.Parameter(torch.zeros(nb_, dp_, d))
        periods = [2.0 ** (i + 1) for i in range(pe_k)] + [2.0 ** pe_k * PHI]
        self.register_buffer("omega", torch.tensor([2 * math.pi / p for p in periods],
                                                   dtype=torch.float32))
        self.amp = nn.Parameter(torch.empty(nb_, dp_, heads, len(periods)).uniform_(0.05, 0.15))
        self.phase = nn.Parameter(torch.zeros(nb_, dp_, heads, len(periods)))
        self.comm1 = nn.Linear(d, d // 2)
        self.comm2 = nn.Linear(d // 2, d)
        nn.init.zeros_(self.comm2.weight)
        nn.init.zeros_(self.comm2.bias)
        self.comm_beta = nn.Parameter(torch.full((nb_,), -2.0))

    def _pe_bias(self, idx, l, T, device):
        wi = torch.arange(T, device=device, dtype=torch.float32)[:, None] * self.omega[None, :]
        ph = self.phase[idx, l]
        am = self.amp[idx, l]
        ai = wi[None, None, :, :] + ph[:, :, None, :]
        Si = am[:, :, None, :] * torch.sin(ai)
        Ci = am[:, :, None, :] * torch.cos(ai)
        Cj = torch.cos(wi)
        Sj = torch.sin(wi)
        return torch.einsum("shtm,jm->shtj", Si, Cj) - torch.einsum("shtm,jm->shtj", Ci, Sj)

    def block_forward(self, x, idx):
        _check_finite(x, "block_forward.x")
        idx = idx.to(x.device)
        S = idx.numel()
        B, T, D = x.shape
        y = x.unsqueeze(0).expand(S, B, T, D)
        for l in range(self.depth):
            n = _ln(y, self.ln1_g[idx, l], self.ln1_b[idx, l])
            nk = n.reshape(S, B * T, D)
            qkv = torch.bmm(nk, self.qkv_w[idx, l]) + self.qkv_b[idx, l][:, None, :]
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(S, B, T, self.heads, self.hd).permute(0, 1, 3, 2, 4)
            k = k.view(S, B, T, self.heads, self.hd).permute(0, 1, 3, 2, 4)
            v = v.view(S, B, T, self.heads, self.hd).permute(0, 1, 3, 2, 4)
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(self.hd)
            scores = scores + self._pe_bias(idx, l, T, x.device)[:, None].to(scores.dtype)
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(causal, -1e4)
            att = torch.softmax(scores, dim=-1)
            o = torch.matmul(att, v.float()).permute(0, 1, 3, 2, 4).reshape(S, B, T, D)
            o = o.to(y.dtype)
            o = torch.bmm(o.reshape(S, B * T, D), self.out_w[idx, l]) + self.out_b[idx, l][:, None, :]
            y = y + o.view(S, B, T, D)
            z = _ln(y, self.ln2_g[idx, l], self.ln2_b[idx, l])
            f = F.gelu(torch.bmm(z.reshape(S, B * T, D), self.ff1_w[idx, l])
                       + self.ff1_b[idx, l][:, None, :])
            f = torch.bmm(f, self.ff2_w[idx, l]) + self.ff2_b[idx, l][:, None, :]
            y = y + f.view(S, B, T, D)
        return y

    def comm(self, Y, g, idx=None, mask=None, trace=None):
        # Legacy path: exactly preserve previous behavior.
        if idx is None and mask is None:
            w = g[:, None, None, None]
            tot = (w * Y).sum(0)
            others = tot.unsqueeze(0) - w * Y
            r = self.comm2(F.gelu(self.comm1(others)))
            beta = torch.sigmoid(self.comm_beta[: Y.shape[0]])[:, None, None, None]
            return Y + beta * w * r

        # New path requires both physical block ids and the (G,B) mask.
        if idx is None or mask is None:
            raise ValueError("comm new path requires both idx and mask")
        if Y.dim() != 4:
            raise ValueError("comm Y must be (G,B,T,D)")
        raw_idx = torch.as_tensor(idx, device=Y.device)
        if raw_idx.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("comm idx must be an integer tensor")
        idx = raw_idx.to(dtype=torch.long)
        mask = torch.as_tensor(mask, device=Y.device, dtype=torch.bool)
        G, B = Y.shape[0], Y.shape[1]
        if idx.shape != (G,) or mask.shape != (G, B):
            raise ValueError(f"comm shape mismatch: Y={tuple(Y.shape)}, idx={tuple(idx.shape)}, mask={tuple(mask.shape)}")
        if (idx < 0).any() or (idx >= self.nb).any():
            raise ValueError(f"comm idx out of range [0,{self.nb}): {idx.tolist()}")
        if torch.unique(idx).numel() != G:
            raise ValueError("comm idx must not contain duplicate physical blocks")
        g_t = torch.as_tensor(g, device=Y.device)
        if g_t.shape != (G,):
            raise ValueError(f"comm g must have shape ({G},), got {tuple(g_t.shape)}")
        w = g_t.to(dtype=Y.dtype, device=Y.device)[:, None, None, None]
        mask4 = mask[:, :, None, None]
        valid_Y = torch.where(mask4, Y, torch.zeros_like(Y))
        _check_valid_finite(valid_Y, mask, "comm.valid_Y")
        weighted_Y = w * valid_Y
        tot = weighted_Y.sum(0)
        others = torch.where(mask4, tot.unsqueeze(0) - weighted_Y, torch.zeros_like(Y))
        _check_valid_finite(others, mask, "comm.others")
        r = self.comm2(F.gelu(self.comm1(others)))
        _check_valid_finite(r, mask, "comm.r")
        beta = torch.sigmoid(self.comm_beta[idx])[:, None, None, None]
        update = torch.where(mask4, beta * w * r, torch.zeros_like(Y))
        out = torch.where(mask4, valid_Y + update, torch.zeros_like(Y))
        _check_valid_finite(out, mask, "comm.out")
        if trace is not None:
            trace.update({"valid_Y": valid_Y, "weighted_Y": weighted_Y, "tot": tot,
                          "others": others, "r": r, "beta": beta,
                          "update": update, "out": out})
        return out


class CrossLayer(nn.Module):
    """m 个查询：query 间因果自注意力 -> 对 n 个编码位置做交叉注意力。"""

    def __init__(self, d, heads, self_causal=True, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.heads, self.hd = heads, d // heads
        self.self_causal = self_causal
        self.ln0 = nn.LayerNorm(d)
        self.sqkv = nn.Linear(d, 3 * d) if self_causal else None
        self.so = nn.Linear(d, d) if self_causal else None
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)

    def forward(self, q, kv):
        B, M, D = q.shape
        if self.self_causal:
            z = self.ln0(q)
            qkv = self.sqkv(z).view(B, M, 3, self.heads, self.hd)
            qq, kk, vv = qkv.unbind(2)
            qq, kk, vv = (t.permute(0, 2, 1, 3) for t in (qq, kk, vv))
            sc = torch.matmul(qq, kk.transpose(-2, -1)) / math.sqrt(self.hd)
            mask = torch.triu(torch.ones(M, M, dtype=torch.bool, device=q.device), 1)
            sc = sc.masked_fill(mask, -1e4)
            att = torch.softmax(sc, -1)
            so = torch.matmul(att, vv).permute(0, 2, 1, 3).reshape(B, M, D)
            q = q + self.so(so)
        n = self.ln1(q)
        qh = self.q(n).view(B, M, self.heads, self.hd).permute(0, 2, 1, 3)
        kh = self.k(kv).view(B, -1, self.heads, self.hd).permute(0, 2, 1, 3)
        vh = self.v(kv).view(B, -1, self.heads, self.hd).permute(0, 2, 1, 3)
        att = torch.softmax(torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.hd), -1)
        o = torch.matmul(att, vh).permute(0, 2, 1, 3).reshape(B, M, D)
        h = q + self.res_scale * self.o(o)
        z = self.ln2(h)
        return h + self.ff2(F.gelu(self.ff1(z)))


class FlowStep(nn.Module):
    def __init__(self, fdim):
        super().__init__()
        self.q = nn.Linear(fdim, fdim)
        self.k = nn.Linear(fdim, fdim)
        self.v = nn.Linear(fdim, fdim)
        self.o = nn.Linear(fdim, fdim)
        self.ln = nn.LayerNorm(fdim)

    def forward(self, h, mask=None, trace=None):
        # Legacy path: exactly preserve previous behavior.
        if mask is None:
            # h: (S,B,T,f)；块之间互相流动（每个 (B,T) 内做一次块间注意力）
            q, k, v = self.q(h), self.k(h), self.v(h)
            att = torch.softmax(torch.einsum("ibtd,jbtd->btij", q, k) / math.sqrt(q.shape[-1]), -1)
            msg = torch.einsum("btij,jbtd->ibtd", att, v)
            return self.ln(h + self.o(msg))

        if h.dim() != 4:
            raise ValueError("FlowStep h must be (G,B,T,f)")
        G, B = h.shape[0], h.shape[1]
        mask = torch.as_tensor(mask, device=h.device, dtype=torch.bool)
        if mask.shape != (G, B):
            raise ValueError(f"FlowStep mask must be (G,B), got {tuple(mask.shape)}")
        count_b = mask.sum(0)
        if (count_b < 1).any():
            raise ValueError("FlowStep empty active set for at least one sample")
        mask4 = mask[:, :, None, None]
        h_in = torch.where(mask4, h, torch.zeros_like(h))
        _check_valid_finite(h_in, mask, "FlowStep.h_in")
        q = torch.where(mask4, self.q(h_in), torch.zeros_like(h_in))
        k = torch.where(mask4, self.k(h_in), torch.zeros_like(h_in))
        v = torch.where(mask4, self.v(h_in), torch.zeros_like(h_in))
        _check_valid_finite(q, mask, "FlowStep.q")
        _check_valid_finite(k, mask, "FlowStep.k")
        _check_valid_finite(v, mask, "FlowStep.v")

        # (G,B,T,f) -> (B,T,G,f)
        Q = q.permute(1, 2, 0, 3)
        K = k.permute(1, 2, 0, 3)
        V = v.permute(1, 2, 0, 3)
        logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(h.shape[-1])
        key_mask = mask.transpose(0, 1)[:, None, None, :]       # (B,1,1,G)
        logits = logits.masked_fill(~key_mask, float("-inf"))
        # Check finite valid logits before softmax; invalid key entries are -inf by design.
        valid_logits = logits.masked_fill(~key_mask, 0.0)
        _check_finite(valid_logits, "FlowStep.logits")
        att = torch.softmax(logits, dim=-1)                     # (B,T,G,G)
        _check_finite(att, "FlowStep.att")
        msg_b = torch.matmul(att, V)                            # (B,T,G,f)
        msg = msg_b.permute(2, 0, 1, 3)                         # (G,B,T,f)
        msg = torch.where(mask4, msg, torch.zeros_like(msg))
        _check_valid_finite(msg, mask, "FlowStep.msg")
        out = self.ln(h_in + self.o(msg))
        out = torch.where(mask4, out, torch.zeros_like(out))
        _check_valid_finite(out, mask, "FlowStep.out")
        if trace is not None:
            trace.update({"h_in": h_in, "q": q, "k": k, "v": v,
                          "Q": Q, "K": K, "V": V, "logits": logits,
                          "att": att, "msg": msg, "out": out,
                          "key_mask": key_mask, "count_b": count_b})
        return out


class InfoFlow(nn.Module):
    """信息流动模型：活跃块提议 -> 低维流动空间 -> T 步动力学 -> settled 场。"""

    def __init__(self, d, fdim=64, steps=2):
        super().__init__()
        self.fdim, self.steps = fdim, steps
        self.proj_in = nn.Linear(d, fdim)
        self.layers = nn.ModuleList([FlowStep(fdim) for _ in range(steps)])
        self.proj_out = nn.Linear(fdim, d)

    def forward(self, delta, mask=None, trace=None):
        # Legacy path: exactly preserve previous behavior.
        if mask is None:
            h = self.proj_in(delta)                    # (S,B,T,f)
            for layer in self.layers:
                h = layer(h)
            mean = h.mean(0)                            # (B,T,f)
            scale = h.norm(dim=-1).mean(0) + 1e-6       # (B,T)
            agree = 1 - ((h - mean.unsqueeze(0)).norm(dim=-1) / scale.unsqueeze(0)).clamp(0, 2)
            cons = self.proj_out(mean)                  # (B,T,D)
            return h, agree, cons

        if delta.dim() != 4:
            raise ValueError("InfoFlow delta must be (G,B,T,D)")
        G, B = delta.shape[0], delta.shape[1]
        mask = torch.as_tensor(mask, device=delta.device, dtype=torch.bool)
        if mask.shape != (G, B):
            raise ValueError(f"InfoFlow mask must be (G,B), got {tuple(mask.shape)}")
        count_b = mask.sum(0)
        if (count_b < 1).any():
            raise ValueError("InfoFlow empty active set for at least one sample")
        mask4 = mask[:, :, None, None]
        delta_in = torch.where(mask4, delta, torch.zeros_like(delta))
        _check_valid_finite(delta_in, mask, "InfoFlow.delta_in")
        h = self.proj_in(delta_in)
        h = torch.where(mask4, h, torch.zeros_like(h))
        _check_valid_finite(h, mask, "InfoFlow.proj_in")
        h_proj = h
        flow_trace = []
        for layer in self.layers:
            step_trace = {} if trace is not None else None
            h = layer(h, mask, trace=step_trace)
            if step_trace is not None:
                flow_trace.append(step_trace)

        valid_h = torch.where(mask4, h, torch.zeros_like(h))
        tot = valid_h.sum(0)
        mean = tot / count_b[:, None, None]
        _check_finite(mean, "InfoFlow.mean")
        valid_norm = torch.where(mask[:, :, None], valid_h.norm(dim=-1), torch.zeros_like(valid_h.norm(dim=-1)))
        scale = valid_norm.sum(0) / count_b[:, None] + 1e-6
        _check_finite(scale, "InfoFlow.scale")
        safe_mean = torch.where(mask4, mean.unsqueeze(0), torch.zeros_like(valid_h))
        centered = valid_h - safe_mean
        agree_raw = 1 - (centered.norm(dim=-1) / scale.unsqueeze(0)).clamp(0, 2)
        agree = torch.where(mask[:, :, None], agree_raw, torch.zeros_like(agree_raw))
        _check_valid_finite(agree, mask, "InfoFlow.agree")
        cons = self.proj_out(mean)
        _check_finite(cons, "InfoFlow.cons")
        if trace is not None:
            trace.update({"proj_in": h_proj, "h": h, "mean": mean, "scale": scale,
                          "agree": agree, "cons": cons, "flow_steps": flow_trace,
                          "count_b": count_b})
        return h, agree, cons


# ---------------------------------------------------------------- Qwen 冻结 IO
class QwenIO(nn.Module):
    """Qwen3-1.7B 的 tokenizer + 冻结 embedding / lm_head，共用一个实例（所有 batch 共享）。

    输入：token ids -> 冻结 embed_w(bf16, V×2048) -> tok_proj(2048->d)
    输出：hidden(d) -> up_proj(d->2048) -> @ 冻结 head_w(bf16, V×2048)
    """

    def __init__(self, embed_path, head_path, hidden=2048, d=256):
        super().__init__()
        from safetensors.torch import load_file
        ew = load_file(embed_path)["model.embed_tokens.weight"]          # bf16 V×2048
        hw = load_file(head_path)["lm_head.weight"]                      # bf16 V×2048
        self.register_buffer("embed_w", ew, persistent=False)
        self.register_buffer("head_w", hw, persistent=False)
        self.vocab, self.hidden = ew.shape
        self.tok_proj = nn.Linear(hidden, d, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        nn.init.normal_(self.tok_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        self._soft = None
        self.refresh_soft()

    @torch.no_grad()
    def refresh_soft(self):
        """refine 软反馈用的 token 嵌入矩阵（V×d），定期刷新以跟上 tok_proj（分块降显存峰值）。"""
        outs = []
        for i in range(0, self.vocab, 8192):
            outs.append(F.linear(self.embed_w[i:i + 8192].float(), self.tok_proj.weight))
        self._soft = torch.cat(outs, 0)

    def embed_ids(self, ids):
        ids = ids.long()
        return self.tok_proj(self.embed_w[ids].float())

    def token_d(self, ids):
        return self.embed_ids(ids)

    def soft_embed(self, probs):
        if self._soft is None or self._soft.device != probs.device:
            self.refresh_soft()
        return torch.matmul(probs, self._soft.to(probs.dtype))

    def logits(self, h):
        up = self.up_proj(h).to(self.head_w.dtype)
        return torch.matmul(up, self.head_w.t()).float()


# ---------------------------------------------------------------- 正式在线模型（v14：E 块 + 权重记忆 M）
class PBSLM(nn.Module):
    """v14：去掉状态向量 M；新增 e_n 个 E 块（高活跃、独立学习率）。

    - E 块与当前选中的 A/C 稠密残差连接（通过每块自己的记忆矩阵 M_e 读出）；
    - E 块之间稠密连接（e_mix 矩阵）；
    - 每个 E 有一块 d×d 的 M_e（requires_grad=False，由 E 写入/擦除，进入 state_dict = 模型记忆）；
    - 主块池仍然 A/C 两角色、互斥、可随时间复用；E 块始终活跃。
    """

    def __init__(self, vocab, k=6, h=2, depth=1, d=256, heads=8, m_max=64,
                 dec_layers_a=2, dec_layers_c=1, pe_k=12,
                 arb="consistency", refine=2, self_causal=True,
                 act_x=0.2, flow_dim=64, flow_steps=2, e_n=10, e_depth=1, io=None):
        super().__init__()
        assert k >= 2 and h >= 0 and refine >= 1
        self.arb, self.refine, self.self_causal = arb, refine, self_causal
        self.vocab, self.k, self.h, self.depth, self.d, self.heads = vocab, k, h, depth, d, heads
        self.m_max, self.act_x = m_max, act_x
        self.e_n = e_n
        self.io = io
        if io is not None:
            self.vocab = io.vocab
        self.tok = nn.Embedding(io.vocab if io is not None else vocab, d) if io is None else None
        self.bank = BlockBank(h + k, depth, d, heads, pe_k)
        self.ebank = BlockBank(e_n, e_depth, d, heads, pe_k)
        # C 隔离：A/C 各自的选块路由（C 的路由在线冻结）
        self.role_a = nn.Linear(d, k, bias=False)
        self.role_c = nn.Linear(d, k, bias=False)
        self.flow = InfoFlow(d, flow_dim, flow_steps)
        # E→A 稠密投影（按原始规格，M 只进 A，不进 C）
        self.e_mix = nn.Parameter(torch.zeros(e_n, e_n))
        self.ea_proj = nn.Linear(d, d)
        nn.init.zeros_(self.ea_proj.weight)
        nn.init.normal_(self.ea_proj.bias, 0, 0.02)
        # 每块 E 的记忆矩阵 M_e（不是状态；由 E 写入，作为模型记忆保存/加载）
        self.mem = nn.ParameterList([nn.Parameter(torch.zeros(d, d), requires_grad=False)
                                     for _ in range(e_n)])
        self.ln_a, self.ln_c, self.ln_final = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q_emb = nn.Parameter(torch.randn(m_max, d) * 0.02)
        self.q_pos = nn.Parameter(torch.randn(m_max, d) * 0.02)
        # C 的独立冻结副本（在线不更新），保证 C 快照不随 A 漂移
        self.q_emb_c = nn.Parameter(torch.randn(m_max, d) * 0.02)
        self.q_pos_c = nn.Parameter(torch.randn(m_max, d) * 0.02)
        self.dec_a = nn.ModuleList([CrossLayer(d, heads, self_causal) for _ in range(dec_layers_a)])
        self.dec_c = nn.ModuleList([CrossLayer(d, heads, self_causal) for _ in range(dec_layers_c)])
        self.refine_proj = nn.Linear(d, d)
        nn.init.zeros_(self.refine_proj.weight)
        nn.init.zeros_(self.refine_proj.bias)
        self.refine_proj_c = nn.Linear(d, d)
        nn.init.zeros_(self.refine_proj_c.weight)
        nn.init.zeros_(self.refine_proj_c.bias)
        if io is None:
            self.head = nn.Linear(d, vocab, bias=False)
            self.head.weight = self.tok.weight
        else:
            self.head = None
        self.a_pred = nn.Linear(d, d)
        self.gate_bias = nn.Parameter(torch.tensor(0.0))

    # ---- 脑活跃度 -> 启用主块数 m_t 与 A/C 配额
    def quotas(self, act=None):
        if act is None:
            act = self.act_x
        cap = max(3, int(round(self.act_x * self.k)))
        m_t = max(3, min(int(round(float(act) * self.k)), cap, self.k))
        kA = max(1, m_t // 2)
        kC = max(1, m_t - kA)
        return kA, kC

    def _select(self, score, kA, kC):
        chosen, g_role = [], torch.zeros(2, self.k, device=score.device)
        taken = torch.zeros(self.k, dtype=torch.bool, device=score.device)
        for r, cnt in enumerate((kA, kC)):
            s = score[r].masked_fill(taken.clone(), -1e9)
            vals, idx = torch.topk(s, cnt)
            g_role = g_role.clone()
            g_role[r, idx] = torch.softmax(vals, 0)
            taken = taken.clone()
            taken[idx] = True
            chosen.append(idx.tolist())
        return chosen, g_role

    def e_forward(self, x):
        """E 块：全部 10 块始终活跃；E-E 稠密连接；每块读出自己的 M_e。"""
        B, T, D = x.shape
        E_out = self.ebank.block_forward(x, torch.arange(self.e_n, device=x.device))
        E_out = self.ebank.comm(E_out, torch.ones(self.e_n, device=x.device))
        mix = torch.einsum("ebtd,ef->fbtd", E_out, self.e_mix)
        E_mix = E_out + mix
        reads = torch.stack([torch.einsum("btd,dv->btv", E_mix[i], self.mem[i])
                             for i in range(self.e_n)], dim=0)          # (E,B,T,D)
        r = reads.mean(0)
        return E_mix, r

    def encode(self, ids, act=None):
        ids = ids.long() if ids.dtype != torch.long else ids
        x = self.io.embed_ids(ids) if self.io is not None else self.tok(ids)
        B, T, D = x.shape
        for i in range(self.h):
            x = self.bank.block_forward(x, torch.tensor([i], device=x.device))[0]
        x_pre = x
        kA, kC = self.quotas(act)
        hmean = x.mean(1).mean(0)
        score_a = self.role_a(hmean)
        score_c = self.role_c(hmean)
        if self.training:
            score_a = score_a + 0.5 * torch.randn_like(score_a)
            score_c = score_c + 0.5 * torch.randn_like(score_c)
        taken = torch.zeros(self.k, dtype=torch.bool, device=x.device)
        chosen, g_role = [], torch.zeros(2, self.k, device=x.device)
        idx_a = torch.topk(score_a, kA)[1]
        taken = taken.clone(); taken[idx_a] = True
        chosen.append(idx_a.tolist())
        s_c = score_c.masked_fill(taken, -1e9)
        idx_c = torch.topk(s_c, kC)[1]
        chosen.append(idx_c.tolist())
        active = sorted(set(sum(chosen, [])))
        ids_par = [self.h + j for j in active]
        Y = self.bank.block_forward(x, torch.tensor(ids_par, device=x.device))
        delta = Y - x.unsqueeze(0)
        delta = self.bank.comm(delta, torch.ones(len(ids_par), device=x.device))
        settled, agree, _ = self.flow(delta)
        row = {j: i for i, j in enumerate(active)}

        def side_cons(role, score):
            rows = [row[j] for j in chosen[role]]
            kk = len(chosen[role])
            # 近均匀的可微权重：主体保持 mean 聚合，10% 由 router 学（避免 softmax 峰值丢信息）
            w = 0.9 * torch.full((kk,), 1.0 / kk, device=score.device) \
                + 0.1 * torch.softmax(score[chosen[role]], dim=0)
            return self.flow.proj_out((settled[rows] * w[:, None, None, None]).sum(0))

        def side_I(role):
            rows = [row[j] for j in chosen[role]]
            return agree[rows].mean(0)

        E_feats, mem_read = self.e_forward(x)
        a_enc = self.ln_a(x_pre + side_cons(0, score_a) + self.ea_proj(mem_read))
        c_enc = self.ln_c(x_pre + side_cons(1, score_c))          # C 不读 M（原始规格：M 只进 k1/A）
        return {"a_enc": a_enc, "c_enc": c_enc, "chosen": chosen, "g_role": g_role,
                "active": active, "I_a": side_I(0), "I_c": side_I(1),
                "e_feats": E_feats, "mem_read": mem_read}

    def _decode(self, layers, q, kv):
        h = q
        for layer in layers:
            h = layer(h, kv)
        z = self.ln_final(h)
        return h, (self.io.logits(z) if self.io is not None else self.head(z))

    def _decode_side(self, layers, q0, kv, proj):
        h = logits = None
        qq = q0
        for r in range(self.refine):
            h, logits = self._decode(layers, qq, kv)
            if r + 1 < self.refine:
                probs = torch.softmax(logits.detach(), -1)
                soft = self.io.soft_embed(probs) if self.io is not None else probs @ self.tok.weight
                qq = qq + proj(soft)
        return h, logits

    def _decode_refine(self, layers, q0, kv, rounds, proj):
        qq = q0
        h = logits = None
        for r in range(rounds):
            h, logits = self._decode(layers, qq, kv)
            if r + 1 < rounds:
                probs = torch.softmax(logits.detach(), -1)
                soft = self.io.soft_embed(probs) if self.io is not None else probs @ self.tok.weight
                qq = qq + proj(soft)
        return logits

    def forward(self, ids_in, m, state=None, tf=None):
        act = None if state is None else state.get("act")
        enc = self.encode(ids_in, act)
        B = ids_in.shape[0]
        q0 = (self.q_emb[:m] + self.q_pos[:m]).unsqueeze(0).expand(B, m, self.d)
        q0_c = (self.q_emb_c[:m] + self.q_pos_c[:m]).unsqueeze(0).expand(B, m, self.d)
        h_a, logits_a = self._decode_side(self.dec_a, q0, enc["a_enc"], self.refine_proj)
        h_c, logits_c = self._decode_side(self.dec_c, q0_c, enc["c_enc"], self.refine_proj_c)
        I_a = enc["I_a"].mean(1)
        I_c = enc["I_c"].mean(1)
        g = torch.sigmoid(8.0 * (I_a - I_c) + self.gate_bias).unsqueeze(-1)
        logits_arb = g.unsqueeze(-1) * logits_a + (1 - g.unsqueeze(-1)) * logits_c
        out = {"a": logits_a, "c": logits_c, "arb": logits_arb, "g": g,
               "cons_a": enc["I_a"], "cons_c": enc["I_c"], "h_a": h_a,
               "state": {}, "e_feats": enc["e_feats"], "mem_read": enc["mem_read"],
               "aux": {"chosen": enc["chosen"], "quotas": (kA := (len(enc["chosen"][0]), len(enc["chosen"][1]))),
                       "active": len(enc["active"]), "act": act}}
        if tf is not None:
            out["verify"] = self._teacher_forced(enc, tf)
        return out

    def teacher_forcing(self, enc, tokens):
        return self._teacher_forced(enc, tokens)

    def _teacher_forced(self, enc, tokens):
        B_tf, m_tf = tokens.shape
        q0 = (self.q_emb[:1] + self.q_pos[:1]).unsqueeze(0).expand(B_tf, 1, self.d)
        tok_d = self.io.token_d(tokens[:, :-1]) if self.io is not None else self.tok(tokens[:, :-1])
        qt = tok_d + self.q_pos[1:m_tf].unsqueeze(0)
        q = torch.cat([q0, qt], dim=1)
        q0c = (self.q_emb_c[:1] + self.q_pos_c[:1]).unsqueeze(0).expand(B_tf, 1, self.d)
        tok_dc = self.io.token_d(tokens[:, :-1]) if self.io is not None else self.tok(tokens[:, :-1])
        qtc = tok_dc + self.q_pos_c[1:m_tf].unsqueeze(0)
        qc = torch.cat([q0c, qtc], dim=1)
        la = self._decode_refine(self.dec_a, q, enc["a_enc"], self.refine, self.refine_proj)
        lc = self._decode_refine(self.dec_c, qc, enc["c_enc"], self.refine, self.refine_proj_c)
        I_a, I_c = enc["I_a"].mean(1), enc["I_c"].mean(1)
        gv = torch.sigmoid(8.0 * (I_a - I_c) + self.gate_bias).unsqueeze(-1)
        return gv.unsqueeze(-1) * la + (1 - gv.unsqueeze(-1)) * lc

    def target_embed(self, ids):
        return self.io.token_d(ids) if self.io is not None else self.tok(ids)

    # ---- 记忆写入 / 擦除（E 负责；不参与反向传播）
    @torch.no_grad()
    def write_memory(self, e_feats, values, gate=1.0, lr=0.05):
        """e_feats: (E,B,T,D)，values: (B,T,D)。每个 E 写自己的 M_e（外积，按 batch 平均）。"""
        k = F.normalize(e_feats.mean(2), dim=-1)       # (E,B,D)
        v = F.normalize(values.mean(1), dim=-1)        # (B,D)
        for i in range(self.e_n):
            outer = torch.einsum("bd,bv->dv", k[i], v)
            self.mem[i].add_(gate * lr * outer)

    @torch.no_grad()
    def decay_memory(self, decay=0.98):
        for i in range(self.e_n):
            self.mem[i].mul_(decay)

    @torch.no_grad()
    def erase_memory(self, idx=None):
        for i in (range(self.e_n) if idx is None else idx):
            self.mem[i].zero_()

    def memory_norms(self):
        return [float(m.norm()) for m in self.mem]

    # ---- 参数分组
    def e_params(self):
        return list(self.ebank.parameters()) + list(self.ea_proj.parameters()) + [self.e_mix]

    def online_params(self):
        ps = list(self.dec_a.parameters())
        ps += list(self.a_pred.parameters()) + list(self.role_a.parameters())
        ps += [self.q_emb, self.q_pos, self.gate_bias]
        ps += list(self.refine_proj.parameters())
        return ps

    def online_params_gate(self):
        return [self.gate_bias] + list(self.role.parameters())

    def frozen_for_online(self):
        # 在线：A 侧 + E（E 是独立学习率的常驻学习通路）；C / 主块 / 流动冻结
        keep = {id(p) for p in self.online_params()} | {id(p) for p in self.e_params()}
        for p in self.parameters():
            if p.requires_grad:
                p.requires_grad_(id(p) in keep)

    def param_groups(self):
        total = sum(p.numel() for p in self.parameters())
        online = sum(p.numel() for p in self.online_params())
        e = sum(p.numel() for p in self.e_params())
        mem = sum(p.numel() for p in self.mem)
        bank = sum(p.numel() for p in self.bank.parameters())
        return {"total": total, "online": online, "e": e, "mem": mem, "bank": bank,
                "dec_a": sum(p.numel() for p in self.dec_a.parameters()),
                "dec_c": sum(p.numel() for p in self.dec_c.parameters())}
