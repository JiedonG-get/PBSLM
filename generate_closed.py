"""连续生成循环（部署口径）：用模型自己生成的 token 续写，验证式多 token 落地。

和 adaptive.py 的区别：adaptive.py 是 teacher-forcing 评测（上下文用 ground truth 续），
这里上下文只来自模型自己的输出（滑窗 n_ctx），是真部署循环。

用法：
  # 小模型（聊天记录语料）
  python generate_closed.py --ckpt ckpt_v12b_d1_n96m32_tv_ratio_tf00.pt
  # zhwiki 目标模型
  python generate_closed.py --ckpt ckpt_v12b_d6_n96m32_tv_ratio_a_side_zhwiki.pt \
      --data-file E:/PycharmProject/data/zhwiki_subset.txt --data-format text
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive import build, commit_block, commit_verify, verify
from run_online import load_chat

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent


@torch.no_grad()
def generate(model, prompt, max_new, mode, tau, n_ctx=96, m=32, act=0.2,
             conf_head=None, commit_cap=0, interrupt_after=0, interrupt_ctx=None,
             interrupt_mem_decay=1.0):
    ctx = prompt.clone()
    out = []
    calls = 0
    sizes = []
    interrupted_at = None
    while len(out) < max_new:
        idx = ctx[-n_ctx:].unsqueeze(0).to(DEVICE)
        want = min(m, max_new - len(out))
        o = model(idx, want, {"act": act})
        probs_p = torch.softmax(o["arb"].float(), -1)
        cand = probs_p.argmax(-1)
        if mode == "single":
            c = 1
        elif mode == "block":
            c = commit_block(probs_p, tau)
        elif mode == "conf":
            conf = torch.sigmoid(conf_head(o["h_a"].float())).squeeze(-1)[0]  # (m,)
            c = 0
            for v in conf.tolist():
                if v >= tau:
                    c += 1
                else:
                    break
            c = max(1, c)
        else:
            pv = verify(model, idx, cand, act)
            c = commit_verify(pv, probs_p, cand, tau)
        c = min(c, want)
        if commit_cap > 0:
            c = min(c, commit_cap)          # 打断粒度上限：最多一次提交这么多 token
        out.extend(cand[0, :c].tolist())
        ctx = torch.cat([ctx, cand[0, :c]], dim=0)
        calls += 1
        sizes.append(c)
        # 打断点：只能在 commit 边界；切换上下文（模拟用户打断/插入新话题）
        if (interrupt_after > 0 and interrupted_at is None
                and len(out) >= interrupt_after):
            interrupted_at = len(out)
            if interrupt_ctx is not None:
                ctx = interrupt_ctx.clone()
            if interrupt_mem_decay < 1.0:
                model.decay_memory(interrupt_mem_decay)
    return out, calls, sizes, interrupted_at


def prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def distinct_ratio(xs, n):
    if len(xs) < n:
        return 0.0
    grams = [tuple(xs[i:i + n]) for i in range(len(xs) - n + 1)]
    return len(set(grams)) / len(grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT_DIR / "ckpt_v12b_d6_n96m32_tv_ratio_a_side_zhwiki.pt"))
    ap.add_argument("--data-file", default="", help="空=聊天记录；zhwiki 用 E:/PycharmProject/data/zhwiki_subset.txt")
    ap.add_argument("--data-format", default="transcripts")
    ap.add_argument("--n-ctx", type=int, default=96)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--act", type=float, default=0.2)
    ap.add_argument("--conf-head", default="", help="confidence head state path")
    ap.add_argument("--conf-kappas", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--commit-cap", type=int, default=0,
                    help="每次最多提交的 token 数（打断粒度上限，0=不限）")
    ap.add_argument("--interrupt-after", type=int, default=0,
                    help="生成到第 N 个 token 时模拟一次打断并切换上下文")
    ap.add_argument("--interrupt-mem-decay", type=float, default=1.0,
                    help="打断时对 M 的衰减系数（<1 生效，模拟打断后记忆重估）")
    args = ap.parse_args()

    model, cfg = build(args.ckpt, act_x=args.act)
    cfg_d = model.d
    _, held, vocab = load_chat(args.data_file, args.data_format)
    starts = np.linspace(0, len(held) - args.n_ctx - args.max_new - 1,
                         args.windows).astype(int).tolist()
    modes = [("single", 0.0), ("block", 0.05), ("block", 0.10)]
    for tau in (0.0, 0.02, 0.05, 0.10):
        modes.append(("verify", tau))
    conf_head = None
    if args.conf_head:
        import torch.nn as nn
        conf_head = nn.Linear(cfg_d, 1).to(DEVICE)
        conf_head.load_state_dict(torch.load(args.conf_head, map_location=DEVICE))
        conf_head.eval()
        for k in [float(x) for x in args.conf_kappas.split(",")]:
            modes.append(("conf", k))

    rows = []
    for mode, tau in modes:
        toks, calls, first_div, acc0, prefix2 = [], [], [], [], []
        rep1, rep2 = [], []
        for s in starts:
            prompt = held[s:s + args.n_ctx].to(DEVICE)
            ref = held[s + args.n_ctx:s + args.n_ctx + args.max_new].tolist()
            gen, c, sizes, __ = generate(model, prompt, args.max_new, mode, tau,
                                         n_ctx=args.n_ctx, m=args.m, act=args.act,
                                         conf_head=conf_head,
                                         commit_cap=args.commit_cap)
            toks.append(len(gen) / max(1, c))
            calls.append(c)
            first_div.append(prefix_len(gen, ref))
            acc0.append(1.0 if gen[0] == ref[0] else 0.0)
            prefix2.append(1.0 if prefix_len(gen, ref) >= 2 else 0.0)
            rep1.append(distinct_ratio(gen, 1))
            rep2.append(distinct_ratio(gen, 2))
        rows.append({"mode": mode, "tau": tau,
                     "tokens_per_call": round(float(np.mean(toks)), 2),
                     "first_div_len": round(float(np.mean(first_div)), 2),
                     "first_token_acc": round(float(np.mean(acc0)), 3),
                     "prefix2_match": round(float(np.mean(prefix2)), 3),
                     "distinct1": round(float(np.mean(rep1)), 3),
                     "distinct2": round(float(np.mean(rep2)), 3),
                     "calls": int(np.mean(calls))})
        r = rows[-1]
        print(f"{r['mode']:6s} tau={r['tau']:.2f}: tokens/call={r['tokens_per_call']:5.2f} "
              f"first_div={r['first_div_len']:5.2f}  top1@0={r['first_token_acc']:.3f} "
              f"prefix2={r['prefix2_match']:.3f}  distinct1/2={r['distinct1']:.2f}/{r['distinct2']:.2f}")
    # ---- 打断模拟：commit 边界处切上下文，测恢复质量与停止粒度
    if args.interrupt_after > 0:
        rows_int = []
        for mode, tau in (("single", 0.0), ("verify", 0.02)):
            rec_acc, rec_p2, stops, gaps = [], [], [], []
            for s in starts[:4]:
                s2 = (s + args.n_ctx * 2) % (len(held) - args.n_ctx - args.max_new - 1)
                prompt = held[s:s + args.n_ctx].to(DEVICE)
                newctx = held[s2:s2 + args.n_ctx].to(DEVICE)
                ref = held[s2 + args.n_ctx:s2 + args.n_ctx + 8].tolist()
                gen, c, sizes, iat = generate(model, prompt, args.max_new, mode, tau,
                                              n_ctx=args.n_ctx, m=args.m, act=args.act,
                                              commit_cap=args.commit_cap,
                                              interrupt_after=args.interrupt_after,
                                              interrupt_ctx=newctx,
                                              interrupt_mem_decay=args.interrupt_mem_decay)
                if iat is None:
                    continue
                post = gen[iat:iat + 8]
                rec_acc.append(float(np.mean([a == b for a, b in zip(post, ref)])) if post else 0.0)
                p2 = 0
                for a, b in zip(post, ref):
                    if a != b:
                        break
                    p2 += 1
                rec_p2.append(p2)
                stops.append(max(sizes))
                gaps.append(len(gen) - iat)
            rows_int.append({"mode": mode, "tau": tau,
                             "recovery_acc@8": round(float(np.mean(rec_acc)), 3),
                             "recovery_prefix": round(float(np.mean(rec_p2)), 2),
                             "max_commit": int(np.max(stops)) if stops else 0,
                             "post_interrupt_tokens": int(np.mean(gaps)) if gaps else 0})
            r = rows_int[-1]
            print(f"[interrupt] {r['mode']:6s} tau={r['tau']:.2f}: "
                  f"max_commit={r['max_commit']} recovery_acc@8={r['recovery_acc@8']:.3f} "
                  f"recovery_prefix={r['recovery_prefix']:.2f}")
        json.dump({"rows": rows, "interrupt": rows_int}, open(OUT_DIR / "closed_loop_results.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("saved closed_loop_results.json (with interrupt)")
        return
    json.dump({"ckpt": args.ckpt, "data": args.data_file, "n_ctx": args.n_ctx,
               "m": args.m, "max_new": args.max_new, "windows": args.windows, "rows": rows},
              open(OUT_DIR / "closed_loop_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved closed_loop_results.json")


if __name__ == "__main__":
    main()
