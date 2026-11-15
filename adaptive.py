"""自适应多 token 输出：自信时一次落地多个 token。

机制（两档）：
  propose：模型一次前向产出 m 个 token 的提议（现有 n→m 解码）。
  commit-block：按提议自身的逐 token 最大概率，取「连续 ≥ 阈值」的最长前缀落地（至少 1 个）。
  commit-verify：把提议的 token 作为查询做一次 teacher-forced 验证（同一解码器、query 间因果），
                 得到 p(t_i | context, t_<i)；只接受「预测正确且概率 ≥ 阈值」的最长前缀；
                 全被拒时退回 1 个 token。

评测（teacher-forcing 口径，不测 wall-clock 延迟）：
  - 每次调用落地的 token 数（tokens/call）
  - 落地 token 的 top-1 准确率、CE（对 ground truth）
  - 不同阈值 τ 下的 tokens/call 与准确率权衡
用法：python adaptive.py --ckpt ckpt_v12b_d6_n96m32_tv_ratio_v13long.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import PBSLM
from run_online import load_chat

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent
CHAT = Path("E:/PycharmProject/transcripts")
N_IN = 96
N_OUT = 32
TAUS = [0.0, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.20]


def load_data():
    files = sorted(CHAT.glob("*.md"))
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(len(text) * 0.85)
    return ids[:cut], ids[cut:], vocab


def build(ckpt_path, act_x=None):
    ck = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    cfg = ck["config"]
    state = ck["state"]
    if act_x is None:
        act_x = cfg.get("act_x", 0.2)
    model = PBSLM(state["tok.weight"].shape[0], k=cfg["k"], h=cfg["h"], depth=cfg["depth"],
                  m_max=cfg.get("m_max", 128), arb=cfg.get("arb", "consistency"),
                  refine=cfg.get("refine", 2),
                  self_causal=not cfg.get("no_self_causal", False),
                  flow_dim=cfg.get("flow_dim", 64), flow_steps=cfg.get("flow_steps", 2),
                  act_x=act_x).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


@torch.no_grad()
def propose(model, ctx, m, act):
    out = model(ctx, m, {"act": act})
    probs = torch.softmax(out["arb"].float(), -1)       # (1,m,V)
    return out, probs


@torch.no_grad()
def verify(model, ctx, cand, act):
    """teacher-forced 验证：query=[q0, emb(t1),...,emb(t_{m-1})]，因果 self-attn 后
    第 k 个输出预测第 k+1 个 token。返回对应 p(t_i) 和 argmax。"""
    enc = model.encode(ctx, act)
    logits = model.teacher_forcing(enc, cand)
    probs = torch.softmax(logits.float(), -1)
    return probs


def commit_block(probs, tau):
    if tau <= 0:                                        # 不设阈值：整块落地
        return probs.shape[1]
    p = probs.max(-1).values[0]                          # (m,)
    c = 0
    for v in p.tolist():
        if v >= tau:
            c += 1
        else:
            break
    return max(1, c)


def commit_verify(probs_v, probs_p, cand, tau):
    """接受「验证预测 == 提议 token 且 p >= tau」的最长前缀；全拒则 1 个。"""
    m = cand.shape[1]
    pv = probs_v[0]                                     # (m,V)
    pp = probs_p[0]
    c = 0
    for i in range(m):
        pred = int(pv[i].argmax())
        prob = float(pp[i, int(cand[0, i])])
        if pred == int(cand[0, i]) and prob >= tau:
            c += 1
        else:
            break
    return max(1, c)


def run_eval(model, held, method, tau, calls=4, windows=16):
    starts = np.linspace(0, len(held) - N_IN - N_OUT * calls - 1, windows).astype(int).tolist()
    tok_per_call, accs, ces, accepted = [], [], [], []
    for s in starts:
        ctx = held[s:s + N_IN].unsqueeze(0).to(DEVICE)
        truth = held[s + N_IN:s + N_IN + N_OUT * calls]
        pos = 0
        for _ in range(calls):
            out, probs_p = propose(model, ctx, N_OUT, 0.2)
            cand = out["arb"].argmax(-1)                # (1,m)
            if method == "block":
                c = commit_block(probs_p, tau)
            elif method == "single":
                c = 1
            else:
                probs_v = verify(model, ctx, cand, 0.2)
                c = commit_verify(probs_v, probs_p, cand, tau)
            c = min(c, N_OUT)
            pred = cand[0, :c]
            tgt = truth[pos:pos + c].to(DEVICE)
            tok_per_call.append(c)
            accs.append(float((pred == tgt).float().mean()))
            p_tok = probs_p[0, torch.arange(c), pred]
            ces.append(float(-torch.log(p_tok.clamp_min(1e-9)).mean()))
            accepted.append(c)
            ctx = torch.cat([ctx, truth[pos:pos + c].unsqueeze(0).to(DEVICE)], dim=1)
            pos += c
            if pos + N_OUT >= truth.numel():
                break
    return {"method": method, "tau": tau, "tokens_per_call": round(float(np.mean(tok_per_call)), 2),
            "acc": round(float(np.mean(accs)), 4), "ce": round(float(np.mean(ces)), 4),
            "calls": len(tok_per_call)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT_DIR / "ckpt_v12b_d6_n96m32_tv_ratio_v13long.pt"))
    ap.add_argument("--data-file", default="")
    ap.add_argument("--data-format", default="transcripts")
    args = ap.parse_args()
    model, cfg = build(args.ckpt)
    _, held, vocab = load_chat(args.data_file, args.data_format)
    print(f"ckpt={Path(args.ckpt).name}  k={cfg['k']} depth={cfg['depth']} "
          f"act_x={cfg.get('act_x')}  n={N_IN} m={N_OUT}")
    results = []
    for method in ("single", "block", "verify"):
        taus = [0.0] if method in ("single",) else TAUS
        for tau in taus:
            r = run_eval(model, held, method, tau)
            results.append(r)
            print(f"{r['method']:6s} tau={r['tau']:.2f}: tokens/call={r['tokens_per_call']:5.2f} "
                  f"top1_acc={r['acc']:.3f}  CE={r['ce']:.3f}  calls={r['calls']}")
    json.dump(results, open(OUT_DIR / "adaptive_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved adaptive_results.json")


if __name__ == "__main__":
    main()
