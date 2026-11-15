"""v12 正式在线模型：可调深度 + n→m 输出 + 并行块内堆叠 —— 训练与长期在线测试。

和旧模型的关键变化：
  - 一次 forward 输入 n 个 token，输出 m 个 token 的分布（非自回归，m 可配置）；
  - 每个真块内部有 depth 个子层（独立权重），一个超参共同调节所有块的深度；
  - 本文件是正式在线模型的训练入口（模型定义在 model.py）。

跑法：
  python run_online.py --depth 1 --n-in 96 --n-out 32
  python run_online.py --depth 2 --n-in 96 --n-out 32
"""
import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import PBSLM, QwenIO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent
CHAT_DIR = Path("E:/PycharmProject/transcripts")
SEQ = 128


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ids_bin(ids_path, vocab_path="", dtype=np.uint16):
    """大语料 memmap：uint16（字符）或 uint32（Qwen token）；按 85/15 切分。"""
    import json
    vocab = None
    if vocab_path:
        vocab = json.load(open(vocab_path, encoding="utf-8"))
    arr = np.memmap(ids_path, dtype=dtype, mode="r")
    n = len(arr)
    cut = int(n * 0.85)
    train = torch.from_numpy(np.asarray(arr[:cut]))
    held = torch.from_numpy(np.asarray(arr[cut:]))
    log(f"ids_bin={ids_path} dtype={np.dtype(dtype).name} bytes={n*arr.itemsize/1e6:.1f}MB "
        f"vocab={len(vocab) if vocab else 'n/a'} train={len(train)} held={len(held)}")
    return train, held, vocab


def load_chat(data_file="", data_format="transcripts"):
    """数据集：默认聊天记录；也支持 --data-file 指向的长文本 / wiki jsonl。"""
    from pathlib import Path as _P
    if data_file:
        fp = _P(data_file)
        assert fp.exists(), f"data file not found: {fp}"
        if data_format == "text":
            text = fp.read_text(encoding="utf-8", errors="ignore")
        else:  # jsonl: WikiExtractor 格式 {"text": ...} 或 {"content": ...}
            import json
            parts = []
            with open(fp, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    t = d.get("content") or d.get("text") or ""
                    if t:
                        parts.append(t)
            text = "\n\n".join(parts)
        log(f"data_file={fp} format={data_format} chars={len(text)}")
    else:
        files = sorted(CHAT_DIR.glob("*.md"))
        text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(len(text) * 0.85)
    return ids[:cut], ids[cut:], vocab


def per_token_ce(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                           reduction="none").view_as(y)


@torch.no_grad()
def eval_nm(model, data, n_in, n_out, n_win=16):
    """固定留出窗口：n 个上下文 token -> 预测接下来 m 个 token。"""
    model.eval()
    starts = np.linspace(0, len(data) - n_in - n_out - 1, n_win).astype(int).tolist()
    rows = {"a": [], "c": [], "arb": []}
    for s in starts:
        x = data[s:s + n_in].unsqueeze(0).to(DEVICE).long()
        y = data[s + n_in:s + n_in + n_out].unsqueeze(0).to(DEVICE).long()
        out = model(x, n_out, None)
        for k in ("a", "c", "arb"):
            rows[k].append(float(per_token_ce(out[k], y).mean()))
    return {k: round(float(np.mean(v)), 4) for k, v in rows.items()}, len(starts)


@torch.no_grad()
def shape_check(model, vocab, pairs):
    """验证 n→m：不同 (n,m) 都能一次调用出 (B,m,V)。"""
    res = []
    for n, m in pairs:
        x = torch.randint(0, vocab, (2, n)).to(DEVICE)
        out = model(x, m, None)
        res.append({"n_in": n, "n_out": m,
                    "logits_shape": list(out["a"].shape),
                    "n_tokens_in": n, "n_tokens_out": m,
                    "ok": list(out["a"].shape) == [2, m, vocab]})
    return res


def load_state_compat(model, state):
    """v14 旧 ckpt -> C 隔离版：role 拆成 role_a/role_c，补 C 的 queries/refine 副本。"""
    missing, unexpected = model.load_state_dict(state, strict=False)
    with torch.no_grad():
        if "role.weight" in state and hasattr(model, "role_a"):
            w = state["role.weight"]
            k = model.k
            model.role_a.weight.copy_(w[:k])
            model.role_c.weight.copy_(w[k:2 * k])
        if "q_emb" in state and hasattr(model, "q_emb_c"):
            model.q_emb_c.copy_(model.q_emb)
            model.q_pos_c.copy_(model.q_pos)
        if hasattr(model, "refine_proj_c"):
            model.refine_proj_c.load_state_dict(model.refine_proj.state_dict())
    return missing, unexpected


def param_table(vocab, k, h, m_max):
    rows = []
    for depth in (1, 2, 3):
        m = PBSLM(vocab, k=k, h=h, depth=depth, m_max=m_max)
        g = m.param_groups()
        rows.append({"depth": depth, "blocks": h + k, "block_depth": depth,
                     "total_params": g["total"], "bank_params": g["bank"],
                     "dec_a_params": g["dec_a"], "dec_c_params": g["dec_c"],
                     "online_params": g["online"]})
        del m
    return rows


def impact_score(ce_c, ce_a, logits_c, kind="ratio"):
    """冲击评分的可操作候选（用户尚未定义，这里做可测实现并给出对比）。"""
    rel = (ce_c.detach() / ce_c.detach().mean().clamp_min(1e-6))
    if kind == "ratio":
        return (1 + 0.5 * rel).clamp(0.5, 3.0)
    if kind == "entropy":
        p = torch.softmax(logits_c.detach().float(), -1)
        H = -(p * (p + 1e-9).log()).sum(-1) / math.log(logits_c.shape[-1])
        return (1 + 0.5 * rel * (1 + H)).clamp(0.5, 3.0)
    if kind == "ac_diff":
        d = (ce_a.detach() - ce_c.detach()).abs()
        return (1 + 0.5 * d / d.mean().clamp_min(1e-6)).clamp(0.5, 3.0)
    raise ValueError(kind)


def shock_score(logits_a, logits_c, y, beta=0.5):
    """震撼值：正确答案任意一侧"本来就知道/想到了" -> 低震撼；两侧都不知道 -> 高震撼。"""
    pa = torch.softmax(logits_a.float(), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    pc = torch.softmax(logits_c.float(), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    known = 1 - (1 - pa) * (1 - pc)
    S = -torch.log(known.clamp_min(1e-3))          # [0, ~6.9]
    w = (1 + beta * S.detach()).clamp(0.5, 3.0)
    return S, w


def optimizer_groups(model, lr, e_lr_mult):
    """主参数用 lr；E 块用自己的学习率（e_lr_mult 倍）。"""
    e_ids = {id(p) for p in model.e_params()}
    main = [p for p in model.parameters() if p.requires_grad and id(p) not in e_ids]
    return [{"params": main, "lr": lr},
            {"params": model.e_params(), "lr": lr * e_lr_mult}]


def pretrain(model, train_ids, vocab, n_in, n_out, steps, seed=0, lr=2e-4,
             batch=8, act_x=0.2, shock_beta=0.5, tf_beta=0.0, commit_beta=0.0,
             e_lr_mult=5.0, mem_decay=0.98, mem_lr=0.05, clip=10.0,
             gradnorm=False, gn_every=200, gn_alpha=1.5, gn_beta=0.1,
             save_every=0, save_prefix=""):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(optimizer_groups(model, lr, e_lr_mult), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * e_lr_mult],
                                                total_steps=steps)
    span = n_in + n_out + 1
    curve = []
    t0 = time.time()
    cap = max(3, int(round(act_x * model.k)))
    shock_log, active_log = [], []
    # GradNorm：只自适应 LM 四项；KL 固定权重（避免 KL 抢占所有权重）
    weights = {"arb": 1.0, "a": 0.5, "c": 0.5, "rep": 0.1, "kl": 0.3}
    gn_terms = {"arb", "a", "c"}          # rep/KL 固定，避免极小梯度项把权重顶到 cap
    gn_floor = {"arb": 0.5, "a": 0.3, "c": 0.3}
    gn_cap = {"arb": 2.0, "a": 1.2, "c": 1.2}
    weights["rep"] = 0.1
    W_lm0 = 2.0
    L0 = {}
    gn_params = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        starts = torch.randint(0, len(train_ids) - span - 1, (batch,), generator=gen)
        x = torch.stack([train_ids[s:s + n_in] for s in starts]).to(DEVICE).long()
        y = torch.stack([train_ids[s + n_in:s + n_in + n_out] for s in starts]).to(DEVICE).long()
        # 训练期随机活跃度（探索：让不同块被选中；模型内部再加 Gumbel 噪声）
        m_t = int(torch.randint(3, cap + 1, (1,), generator=gen).item())
        act = m_t / model.k
        tf_on = tf_beta > 0 or commit_beta > 0
        out = model(x, n_out, {"act": act}, tf=y if tf_on else None)
        ce_a = per_token_ce(out["a"].float(), y)
        ce_c = per_token_ce(out["c"].float(), y)
        ce_arb = per_token_ce(out["arb"].float(), y)
        S, w = shock_score(out["a"], out["c"], y, shock_beta)
        rep = ((model.a_pred(out["h_a"]) - model.target_embed(y).detach()) ** 2).mean(-1)
        terms = {"arb": ce_arb.mean(), "a": (ce_a * w).mean(),
                 "c": ce_c.mean(), "rep": (rep * w).mean()}
        if commit_beta > 0 and "verify" in out:
            p_blk = torch.softmax(out["arb"].float(), -1).clamp_min(1e-8)
            p_ver = torch.softmax(out["verify"].float().detach(), -1).clamp_min(1e-8)
            p_ver = p_ver / p_ver.sum(-1, keepdim=True)
            terms["kl"] = (p_blk * (p_blk.log() - p_ver.log())).sum(-1).mean()
        if step == 0:
            L0 = {k: float(v.detach()) for k, v in terms.items()}
        if gradnorm and step > 0 and step % gn_every == 0:
            if not gn_params:
                if model.io is not None:
                    gn_params = [model.io.tok_proj.weight]
                else:
                    gn_params = [model.tok.weight]
                gn_params += list(model.flow.parameters())
                gn_params += [model.role_a.weight, model.role_c.weight]
                gn_params += list(model.dec_c.parameters())
            gn_params = [p for p in gn_params if p.requires_grad]
            Gn = {}
            for k in gn_terms:
                if k not in terms:
                    continue
                gs = torch.autograd.grad(terms[k], gn_params, retain_graph=True, allow_unused=True)
                Gn[k] = float(torch.sqrt(sum((g ** 2).sum() for g in gs if g is not None)))
            Gmean = float(np.mean([v for v in Gn.values() if v > 0])) if any(v > 0 for v in Gn.values()) else 0.0
            if Gmean > 0:
                for k in gn_terms:
                    if k in terms and Gn.get(k, 0) > 1e-12:
                        r = float(terms[k].detach()) / max(L0.get(k, 1e-6), 1e-6)
                        target = Gmean * (r ** gn_alpha)
                        w = weights[k] * (target / Gn[k]) ** gn_beta
                        weights[k] = float(np.clip(w, gn_floor[k], gn_cap[k]))
                # 归一化 LM 权重和，再夹一次边界
                cur = sum(weights[k] for k in gn_terms)
                if cur > 0:
                    for k in gn_terms:
                        weights[k] = float(np.clip(weights[k] * W_lm0 / cur,
                                                   gn_floor[k], gn_cap[k]))
            log(f"  gradnorm w={ {k: round(weights[k],3) for k in sorted(gn_terms)} } "
                f"G={ {k: round(v,1) for k,v in Gn.items()} }")
        loss = sum(weights[k] * terms[k] for k in terms)
        if tf_beta > 0 and "verify" in out:
            loss = loss + tf_beta * per_token_ce(out["verify"].float(), y).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        sched.step()
        # E 负责写记忆：每个 chunk 写入 + 衰减
        model.write_memory(out["e_feats"].detach(), model.target_embed(y).detach(), gate=1.0, lr=mem_lr)
        model.decay_memory(mem_decay)
        if save_every > 0 and step > 0 and step % save_every == 0 and save_prefix:
            sp = save_prefix.replace(".pt", f".step{step}.pt")
            torch.save({"state": model.state_dict(), "step": step,
                        "config": {"k": model.k, "depth": model.depth, "act_x": model.act_x}}, sp)
            log(f"  saved {sp}")
        curve.append(round(float(loss.detach()), 4))
        shock_log.append(round(float(S.mean()), 4))
        active_log.append(out["aux"]["active"])
        if step % 100 == 0:
            ev, _ = eval_nm(model, train_ids[-8000:], n_in, n_out, n_win=4)
            log(f"  pretrain s{step} loss={float(loss.detach()):.3f} shock={S.mean():.3f} "
                f"active={out['aux']['active']} train_probe arb={ev['arb']:.4f}")
    log(f"pretrain done in {time.time()-t0:.1f}s")
    return {"loss_curve": curve, "shock_curve": shock_log, "active_curve": active_log,
            "seconds": round(time.time() - t0, 1)}



@contextmanager
def zero_memory(model, idx):
    saved = model.mem[idx].clone()
    with torch.no_grad():
        model.mem[idx].zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            model.mem[idx].copy_(saved)


@torch.no_grad()
def quick_ce(model, data, n_in, n_out, n_win=4, side="arb"):
    model.eval()
    starts = np.linspace(0, len(data) - n_in - n_out - 1, n_win).astype(int).tolist()
    vals = []
    for s in starts:
        x = data[s:s + n_in].unsqueeze(0).to(DEVICE).long()
        y = data[s + n_in:s + n_in + n_out].unsqueeze(0).to(DEVICE).long()
        out = model(x, n_out, None)
        vals.append(float(per_token_ce(out[side], y).mean()))
    return float(np.mean(vals))


@contextmanager
def zero_memories(model, idxs):
    saved = [model.mem[i].clone() for i in idxs]
    with torch.no_grad():
        for i in idxs:
            model.mem[i].zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for i, sv in zip(idxs, saved):
                model.mem[i].copy_(sv)


def runtime_consolidate(model, replay_buf, held_ids, n_in, n_out, args):
    """周期性运行时固化：
    - 单块重要 -> 固化该块；单块都冗余 -> 用 top-k 分组重要性 -> 固化组；
    - 组也冗余 -> 按单块重要性顺序擦除最冗余的 M（释放容量）；
    - C 快照护栏（C 冻结，前后 C CE 必须一致）+ held CE 恶化回滚；成功后 M 衰减。"""
    norms = model.memory_norms()
    mature = [i for i, n in enumerate(norms) if n >= args.consolidate_min_norm]
    if not mature:
        return {"event": "no_mature", "norms": [round(x, 3) for x in norms]}
    base_arb = quick_ce(model, held_ids, n_in, n_out, args.consolidate_probe, "arb")
    base_c = quick_ce(model, held_ids, n_in, n_out, args.consolidate_probe, "c")
    ind = {}
    for i in mature:
        with zero_memories(model, [i]):
            ind[i] = round(quick_ce(model, held_ids, n_in, n_out,
                                    args.consolidate_probe, "arb") - base_arb, 5)
    groups = {}
    for k in (3, min(5, len(mature))):
        if 1 < k <= len(mature):
            ids = sorted(mature, key=lambda i: -norms[i])[:k]
            with zero_memories(model, ids):
                groups[k] = round(quick_ce(model, held_ids, n_in, n_out,
                                           args.consolidate_probe, "arb") - base_arb, 5)
    # 固化候选：按重要性降序（single 与 group 里选重要性最高且 > 阈值的）
    cands = []
    for i in mature:
        if ind[i] > 1e-4:
            cands.append(([i], ind[i], "single"))
    for k, gv in groups.items():
        if gv > 1e-4:
            cands.append((sorted(mature, key=lambda i: -norms[i])[:k], gv, f"group{k}"))
    if getattr(args, "consolidate_policy", "important_first") == "redundant_first":
        cands = [(ids, -imp, kind) for ids, imp, kind in cands]
    cand = max(cands, key=lambda x: x[1]) if cands else None
    if cand is None and getattr(args, "consolidate_force", False) and len(mature) >= 2:
        k = min(3, len(mature))
        cand = (sorted(mature, key=lambda i: -norms[i])[:k], 0.0, f"forced_group{k}")
    if cand is None:
        # 冗余升序擦除：最冗余（单块重要性最低）先擦
        erase = min(mature, key=lambda i: ind[i])
        with torch.no_grad():
            model.mem[erase].mul_(0.5)
        return {"event": "erase_redundant", "e": int(erase), "individual": ind,
                "groups": groups, "mem_norm": round(float(model.mem[erase].norm()), 2)}
    ids, importance, kind = cand
    with zero_memories(model, ids):
        base_noM = quick_ce(model, held_ids, n_in, n_out, args.consolidate_probe, "arb")
    A = model.online_params()
    saved = [p.detach().clone() for p in A]
    opt = torch.optim.AdamW(A, lr=args.consolidate_lr)
    model.eval()
    for step in range(args.consolidate_steps):
        x, y = replay_buf[np.random.randint(len(replay_buf))]
        model.zero_grad(set_to_none=True)
        t_out = model(x, n_out, None)
        p_t = torch.softmax(t_out["arb"].float().detach(), -1).clamp_min(1e-8)
        with zero_memories(model, ids):
            s_out = model(x, n_out, None)
            s_ce = per_token_ce(s_out["arb"].float(), y).mean()
            p_s = torch.softmax(s_out["arb"].float(), -1).clamp_min(1e-8)
            loss = s_ce + (p_t * (p_t.log() - p_s.log())).sum(-1).mean()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(A, 1.0)
        opt.step()
    new_arb = quick_ce(model, held_ids, n_in, n_out, args.consolidate_probe, "arb")
    new_c = quick_ce(model, held_ids, n_in, n_out, args.consolidate_probe, "c")
    with zero_memories(model, ids):
        new_noM = quick_ce(model, held_ids, n_in, n_out, args.consolidate_probe, "arb")
    c_guard = abs(new_c - base_c) < 1e-6
    accepted = ((new_noM <= base_noM - 1e-4)
                and (new_arb <= base_arb + args.consolidate_held_delta) and c_guard)
    if accepted:
        with torch.no_grad():
            for i in ids:
                model.mem[i].mul_(0.7)
    else:
        with torch.no_grad():
            for p, sv in zip(A, saved):
                p.copy_(sv)
    return {"event": "keep" if accepted else "rollback", "kind": kind,
            "policy": getattr(args, "consolidate_policy", "important_first"),
            "e": [int(i) for i in ids], "importance": importance, "individual": ind,
            "groups": groups, "arb_before": round(base_arb, 4),
            "arb_after": round(new_arb, 4),
            "noM_before": round(base_noM, 4), "noM_after": round(new_noM, 4),
            "c_guard": bool(c_guard)}


def online_stream(model, train_ids, held_ids, n_in, n_out, n_chunks, lr=3e-4,
                  eval_every=500, decay=False, shock_beta=0.5,
                  scope="a_side", replay=0, eval_windows=16, act_x=0.2,
                  e_lr_mult=5.0, mem_decay=0.98, mem_lr=0.05, mem_write_shock=1.0,
                  consolidate_every=0, args_cons=None):
    consolidate_act_frac = getattr(args_cons, "consolidate_act_frac", 0.7) if args_cons else 0.7
    model.frozen_for_online()
    if scope == "a_side":
        trainable = model.online_params() + model.e_params()
        opt = torch.optim.AdamW(optimizer_groups(model, lr, e_lr_mult))
    else:
        trainable = model.online_params_gate()
        opt = torch.optim.AdamW(trainable, lr=lr)
    replay_buf = []
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_chunks),
                                                       eta_min=lr * 0.1) if decay else None
    span = len(train_ids) - n_in - n_out - 1
    rec = {k: [] for k in ("ce_a", "ce_c", "ce_arb", "rep", "g", "shock", "act",
                           "active", "mem_norm")}
    held_curve = []
    consolidate_events = []
    state = {"act": act_x}
    t0 = time.time()
    model.eval()
    for k in range(n_chunks):
        s = (k * n_in) % span
        x = train_ids[s:s + n_in].unsqueeze(0).to(DEVICE).long()
        y = train_ids[s + n_in:s + n_in + n_out].unsqueeze(0).to(DEVICE).long()
        with torch.no_grad():
            out = model(x, n_out, state)
            rec["ce_a"].append(float(per_token_ce(out["a"], y).mean()))
            rec["ce_c"].append(float(per_token_ce(out["c"], y).mean()))
            rec["ce_arb"].append(float(per_token_ce(out["arb"], y).mean()))
            rec["g"].append(float(out["g"].mean()))
            S, _ = shock_score(out["a"], out["c"], y, shock_beta)
            I = 0.5 * (out["cons_a"].mean() + out["cons_c"].mean())
            next_act = act_x * torch.sigmoid(1.0 * (S.mean() - 1.5) + 2.0 * (1 - I) - 1.0)
            state = {"act": float(next_act)}
            rec["shock"].append(round(float(S.mean()), 4))
            rec["act"].append(round(float(next_act), 4))
            rec["active"].append(out["aux"]["active"])
            rec["mem_norm"].append(round(float(sum(model.memory_norms()) / model.e_n), 3))
        opt.zero_grad(set_to_none=True)
        out = model(x, n_out, state)
        ce_c = per_token_ce(out["c"].float(), y)
        ce_a = per_token_ce(out["a"].float(), y)
        S, w = shock_score(out["a"], out["c"], y, shock_beta)
        rep = ((model.a_pred(out["h_a"]) - model.target_embed(y).detach()) ** 2).mean(-1)
        loss = ((0.5 * ce_a + rep) * w).mean()
        if replay > 0 and replay_buf:
            rx, ry = replay_buf[np.random.randint(len(replay_buf))]
            with torch.no_grad():
                rout = model(rx, n_out, None)
            r_a = per_token_ce(rout["a"].float(), ry)
            _, r_w = shock_score(rout["a"], rout["c"], ry, shock_beta)
            r_rep = ((model.a_pred(rout["h_a"]) - model.target_embed(ry).detach()) ** 2).mean(-1)
            loss = loss + 0.5 * ((0.5 * r_a.detach() + r_rep) * r_w).mean()
        if replay > 0:
            replay_buf.append((x.detach(), y.detach()))
            if len(replay_buf) > replay:
                replay_buf.pop(0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        # E 负责写记忆：高震撼才写（低震撼 = 已知，不写）
        if scope == "a_side":
            g_write = 1.0 if float(S.mean()) >= mem_write_shock else 0.0
            model.write_memory(out["e_feats"].detach(), model.target_embed(y).detach(),
                               gate=g_write, lr=mem_lr)
            model.decay_memory(mem_decay)
        rec["rep"].append(float(rep.detach().mean()))
        # ---- 周期性运行时固化（低活跃 + 高震撼之后才做） ----
        if (consolidate_every > 0 and (k + 1) % consolidate_every == 0
                and replay_buf and len(rec["act"]) >= 50):
            recent_act = float(np.mean(rec["act"][-50:]))
            if recent_act <= consolidate_act_frac * act_x:
                ev = runtime_consolidate(model, replay_buf, held_ids, n_in, n_out, args_cons)
                consolidate_events.append(ev)
                log(f"  [consolidate] chunk {k+1}: {ev}")
            else:
                consolidate_events.append({"event": "skipped_high_activity",
                                           "recent_act": round(recent_act, 3)})
        if (k + 1) % eval_every == 0:
            ev, nw = eval_nm(model, held_ids, n_in, n_out, n_win=eval_windows)
            held_curve.append({"chunk": k + 1, "epoch": round(k / (span / n_in), 3),
                               "held": ev})
            log(f"  online {k+1}/{n_chunks} stream_arb={np.mean(rec['ce_arb'][-eval_every:]):.4f} "
                f"held_arb={ev['arb']:.4f} act={np.mean(rec['act'][-eval_every:]):.3f} "
                f"active={np.mean(rec['active'][-eval_every:]):.1f} "
                f"shock={np.mean(rec['shock'][-eval_every:]):.3f} "
                f"mem={np.mean(rec['mem_norm'][-eval_every:]):.2f}")
    log(f"online done {n_chunks} chunks in {time.time()-t0:.1f}s")
    return {"records": {k: [round(float(v), 4) for v in arr] for k, arr in rec.items()},
            "held_curve": held_curve, "consolidate_events": consolidate_events,
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--h", type=int, default=2)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--n-in", type=int, default=96)
    ap.add_argument("--n-out", type=int, default=32)
    ap.add_argument("--m-max", type=int, default=128)
    ap.add_argument("--arb", choices=["consistency", "scalar"], default="consistency")
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--no-self-causal", action="store_true")
    ap.add_argument("--online-lr", type=float, default=3e-4)
    ap.add_argument("--online-decay", action="store_true")
    ap.add_argument("--cons", choices=["tv", "topk", "cos", "kl"], default="tv")
    ap.add_argument("--impact", choices=["ratio", "entropy", "ac_diff"], default="ratio")
    ap.add_argument("--online-scope", choices=["a_side", "gate_only"], default="a_side")
    ap.add_argument("--replay", type=int, default=0)
    ap.add_argument("--eval-windows", type=int, default=16)
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--skip-pretrain", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--chunks", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--act-x", type=float, default=0.2)
    ap.add_argument("--flow-dim", type=int, default=64)
    ap.add_argument("--flow-steps", type=int, default=2)
    ap.add_argument("--e-n", type=int, default=10)
    ap.add_argument("--e-depth", type=int, default=1)
    ap.add_argument("--e-lr-mult", type=float, default=5.0)
    ap.add_argument("--mem-decay", type=float, default=0.98)
    ap.add_argument("--mem-lr", type=float, default=0.05)
    ap.add_argument("--pretrain-clip", type=float, default=10.0,
                    help="预训练梯度裁剪阈值（复合损失尺度大，1.0 会把更新压掉）")
    ap.add_argument("--mem-write-shock", type=float, default=1.0)
    ap.add_argument("--consolidate-every", type=int, default=0,
                    help="每 N 个 chunk 做一次运行时固化（0=关闭）")
    ap.add_argument("--consolidate-steps", type=int, default=2)
    ap.add_argument("--consolidate-lr", type=float, default=1e-4)
    ap.add_argument("--consolidate-probe", type=int, default=4)
    ap.add_argument("--consolidate-min-norm", type=float, default=0.1)
    ap.add_argument("--consolidate-held-delta", type=float, default=0.02)
    ap.add_argument("--consolidate-act-frac", type=float, default=0.7)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--consolidate-policy", choices=["important_first", "redundant_first"],
                    default="important_first",
                    help="固化顺序：默认重要降序；擦除始终冗余升序")
    ap.add_argument("--consolidate-force", action="store_true",
                    help="测试用：忽略重要性门槛，强制对 top-3 组做固化（验证 keep/rollback 路径）")
    ap.add_argument("--shock-beta", type=float, default=0.5)
    ap.add_argument("--sweep", default="", help="逗号分隔的 n x m 评测对，如 64x16,96x32")
    ap.add_argument("--tf-beta", type=float, default=0.0, help="teacher-forced 验证头损失权重")
    ap.add_argument("--commit-beta", type=float, default=0.0, help="多 token 落地一致性损失权重")
    ap.add_argument("--data-file", default="", help="长语料路径（text 或 wiki jsonl）")
    ap.add_argument("--data-format", choices=["transcripts", "text", "jsonl"], default="transcripts")
    ap.add_argument("--data-ids", default="", help="大语料 memmap ids 路径")
    ap.add_argument("--vocab-json", default="", help="配色 memmap 语料的词表 json（字符级）")
    ap.add_argument("--ids-dtype", choices=["uint16", "uint32"], default="uint16")
    ap.add_argument("--qwen-embed", default="", help="Qwen embed_tokens.safetensors")
    ap.add_argument("--qwen-head", default="", help="Qwen lm_head.safetensors")
    ap.add_argument("--gradnorm", action="store_true", help="多损失自适应权重（GradNorm）")
    ap.add_argument("--gn-every", type=int, default=200)
    ap.add_argument("--gn-alpha", type=float, default=1.5)
    ap.add_argument("--gn-beta", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--save-prefix", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dtype = np.uint32 if args.ids_dtype == "uint32" else np.uint16
    if args.data_ids:
        train_ids, held_ids, vocab = load_ids_bin(args.data_ids, args.vocab_json, dtype)
    else:
        train_ids, held_ids, vocab = load_chat(args.data_file, args.data_format)
    if args.k <= 8 and args.depth <= 2:
        ptab = param_table(len(vocab) if vocab else 151936, args.k, args.h, args.m_max)
    else:
        ptab = []
    for r in ptab:
        log(f"depth={r['depth']} blocks={r['blocks']} total={r['total_params']/1e6:.2f}M "
            f"bank={r['bank_params']/1e6:.2f}M online={r['online_params']/1e6:.2f}M")
    io = None
    if args.qwen_embed:
        io = QwenIO(args.qwen_embed, args.qwen_head, d=256).to(DEVICE)
    model = PBSLM(len(vocab) if vocab else (io.vocab if io else 0),
                  k=args.k, h=args.h, depth=args.depth, m_max=args.m_max,
                  arb=args.arb, refine=args.refine,
                  self_causal=not args.no_self_causal,
                  act_x=args.act_x, flow_dim=args.flow_dim,
                  flow_steps=args.flow_steps, e_n=args.e_n,
                  e_depth=args.e_depth, io=io).to(DEVICE)
    if io is not None:
        io.refresh_soft()
        log(f"qwen io: vocab={io.vocab} hidden={io.hidden} (embed/head frozen, shared)")
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, weights_only=False, map_location=DEVICE)
        try:
            model.load_state_dict(ck["state"])
            log(f"loaded ckpt {args.init_ckpt} (strict)")
        except RuntimeError:
            miss, unexp = load_state_compat(model, ck["state"])
            log(f"loaded ckpt {args.init_ckpt} (compat: missing={len(miss)}, unexpected={len(unexp)})")
    g = model.param_groups()
    log(f"model: k={args.k} h={args.h} depth={args.depth} E={args.e_n}x{args.e_depth} "
        f"total={g['total']/1e6:.2f}M online={g['online']/1e6:.2f}M "
        f"E={g['e']/1e6:.2f}M mem={g['mem']/1e6:.2f}M")
    if args.sweep:
        pairs = [tuple(int(x) for x in p.split("x")) for p in args.sweep.split(",")]
    else:
        pairs = [(32, 8), (64, 16), (96, 32), (128, 32), (192, 32),
                 (256, 64), (96, 8), (96, 64), (96, 128)]
    shapes = shape_check(model, model.vocab, pairs)
    log("shape check: " + json.dumps(shapes, ensure_ascii=False))
    assert all(s["ok"] for s in shapes), "n->m shape check failed"

    if args.skip_pretrain:
        pt = {"loss_curve": [], "seconds": 0.0, "skipped": True}
    else:
        pt = pretrain(model, train_ids, vocab, args.n_in, args.n_out, args.steps,
                      seed=args.seed, batch=args.batch, act_x=args.act_x,
                      shock_beta=args.shock_beta, tf_beta=args.tf_beta,
                      commit_beta=args.commit_beta, e_lr_mult=args.e_lr_mult,
                      mem_decay=args.mem_decay, mem_lr=args.mem_lr,
                      clip=args.pretrain_clip, gradnorm=args.gradnorm,
                      gn_every=args.gn_every, gn_alpha=args.gn_alpha,
                      gn_beta=args.gn_beta, save_every=args.save_every,
                      save_prefix=args.save_prefix)
    held_before, n_win = eval_nm(model, held_ids, args.n_in, args.n_out,
                                 n_win=args.eval_windows)
    log(f"after pretrain held(n={args.n_in},m={args.n_out}) = {held_before}")
    sweep = []
    for (n, m) in pairs:
        ev, nw = eval_nm(model, held_ids, n, m, n_win=args.eval_windows)
        sweep.append({"n_in": n, "n_out": m, **ev, "windows": nw})
        log(f"  n={n} m={m}: held arb={ev['arb']:.4f} a={ev['a']:.4f} c={ev['c']:.4f}")
    before = {"main": held_before, "sweep": sweep}

    if args.chunks > 0:
        on = online_stream(model, train_ids, held_ids, args.n_in, args.n_out, args.chunks,
                           lr=args.online_lr, decay=args.online_decay,
                           shock_beta=args.shock_beta, scope=args.online_scope,
                           replay=args.replay, eval_windows=args.eval_windows,
                           act_x=args.act_x, e_lr_mult=args.e_lr_mult,
                           mem_decay=args.mem_decay, mem_lr=args.mem_lr,
                           mem_write_shock=args.mem_write_shock,
                           consolidate_every=args.consolidate_every,
                           eval_every=args.eval_every, args_cons=args)
    else:
        on = {"records": {}, "held_curve": [], "seconds": 0.0, "skipped": True}
    held_after, _ = eval_nm(model, held_ids, args.n_in, args.n_out,
                            n_win=args.eval_windows)
    sweep_after = []
    for (n, m) in pairs:
        ev, nw = eval_nm(model, held_ids, n, m, n_win=args.eval_windows)
        sweep_after.append({"n_in": n, "n_out": m, **ev, "windows": nw})
    log(f"after online held = {held_after}")

    ckpt = OUT_DIR / f"ckpt_v12b_d{args.depth}_n{args.n_in}m{args.n_out}_{args.cons}_{args.impact}{args.tag}.pt"
    torch.save({"state": model.state_dict(), "config": vars(args)}, ckpt)
    res = {"version": "v12b", "config": vars(args), "checkpoint": str(ckpt),
           "param_table": ptab, "model_params": g,
           "shape_check": shapes,
           "pretrain": pt, "held_before": before, "online": on,
           "held_after": held_after, "sweep_after": sweep_after,
           "gate": None}
    name = (f"results_v12_d{args.depth}_n{args.n_in}m{args.n_out}"
            f"_{args.cons}_{args.impact}_{args.online_scope}{args.tag}.json")
    (OUT_DIR / name).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved {OUT_DIR / name}")


if __name__ == "__main__":
    main()
