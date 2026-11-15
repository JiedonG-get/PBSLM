"""记忆命中 / 消融评估：M 到底有没有帮助，以及每块 E 的记忆贡献。

指标：
  CE(M)        : 正常带记忆
  CE(no all M) : 所有 M_e 置零
  CE(no M_i)   : 只置零第 i 块 E 的 M_e -> 该块记忆的贡献 = CE(no M_i) - CE(M)
另外报告每块 E 的 M 范数、M 读出的方向与目标 token 嵌入的平均余弦。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive import build
from run_online import per_token_ce, load_chat

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent
N_IN, N_OUT = 96, 32


@torch.no_grad()
def eval_ce(model, data, starts, ablate=None):
    model.eval()
    ce = []
    for s in starts:
        x = data[s:s + N_IN].unsqueeze(0).to(DEVICE)
        y = data[s + N_IN:s + N_IN + N_OUT].unsqueeze(0).to(DEVICE)
        out = model(x, N_OUT, {"act": 0.2})
        ce.append(float(per_token_ce(out["arb"], y).mean()))
    return float(np.mean(ce))


def make_starts(n, count):
    return np.linspace(0, n - N_IN - N_OUT - 1, count).astype(int).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT_DIR / "ckpt_v12b_d1_n96m32_tv_ratio_v14a.pt"))
    ap.add_argument("--data-file", default="")
    ap.add_argument("--data-format", default="transcripts")
    ap.add_argument("--windows", type=int, default=16)
    args = ap.parse_args()
    model, cfg = build(args.ckpt, act_x=0.2)
    _, held, vocab = load_chat(args.data_file, args.data_format)
    starts = make_starts(len(held), args.windows)
    print(f"ckpt={Path(args.ckpt).name} E={model.e_n} windows={len(starts)}")

    # 记忆读出与目标的平均余弦
    cos = []
    with torch.no_grad():
        e_idx = 0
        for s in starts:
            x = held[s:s + N_IN].unsqueeze(0).to(DEVICE)
            y = held[s + N_IN:s + N_IN + N_OUT].unsqueeze(0).to(DEVICE)
            enc = model.encode(x, 0.2)
            e = enc["e_feats"][e_idx]                       # (1,T,D)
            r = torch.einsum("btd,dv->btv", e[:, -N_OUT:], model.mem[e_idx])
            v = model.tok(y)
            cos.append(float(F.cosine_similarity(r, v, dim=-1).mean()))

    base = eval_ce(model, held, starts)
    saved = [m.clone() for m in model.mem]
    for m in model.mem:
        m.zero_()
    no_all = eval_ce(model, held, starts)
    for m, s in zip(model.mem, saved):
        m.copy_(s)
    rows = []
    for i in range(model.e_n):
        saved_i = model.mem[i].clone()
        model.mem[i].zero_()
        no_i = eval_ce(model, held, starts)
        model.mem[i].copy_(saved_i)
        rows.append({"e": i, "m_norm": round(float(model.mem[i].norm()), 4),
                     "ce_no_this": round(no_i, 5),
                     "contrib": round(no_i - base, 5)})
    out = {"ckpt": args.ckpt, "ce_with_mem": round(base, 5),
           "ce_no_mem": round(no_all, 5),
           "mem_gain": round(no_all - base, 5),
           "read_target_cos": round(float(np.mean(cos)), 4),
           "per_e": sorted(rows, key=lambda r: -r["contrib"])}
    print(f"CE with M={base:.4f}  no M={no_all:.4f}  mem_gain={no_all-base:+.4f}  "
          f"read-target cos={np.mean(cos):+.3f}")
    print("per-E contrib (CE 升高表示该块记忆有贡献)：")
    for r in out["per_e"]:
        print(f"  E{r['e']:02d} norm={r['m_norm']:.2f} contrib={r['contrib']:+.5f}")
    json.dump(out, open(OUT_DIR / "memory_hit.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved memory_hit.json")


if __name__ == "__main__":
    main()
