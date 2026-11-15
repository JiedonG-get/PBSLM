"""记忆重要性 / 冗余分析：决定每块 E 的固化顺序与擦除顺序。

指标：
  ind_i        : 单独置零 M_e 的 CE 变化（≈0 说明单块冗余）
  group_k      : 置零按范数最大的 k 块，看群体必要性
  shapi_i      : 随机子集（大小 5，含 i）置零的平均 CE 变化 = 分组重要性
  pair_cos     : M_e 两两余弦相似度（冗余结构）
输出：固化顺序（分组重要性降序）与擦除顺序（单块重要性升序）。
"""
import argparse
import itertools
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
def ce_arb(model, data, starts):
    vals = []
    for s in starts:
        x = data[s:s + N_IN].unsqueeze(0).to(DEVICE)
        y = data[s + N_IN:s + N_IN + N_OUT].unsqueeze(0).to(DEVICE)
        out = model(x, N_OUT, {"act": 0.2})
        vals.append(float(per_token_ce(out["arb"], y).mean()))
    return float(np.mean(vals))


@torch.no_grad()
def ce_with_zero(model, data, starts, idxs):
    saved = [model.mem[i].clone() for i in idxs]
    for i in idxs:
        model.mem[i].zero_()
    try:
        return ce_arb(model, data, starts)
    finally:
        for i, sv in zip(idxs, saved):
            model.mem[i].copy_(sv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT_DIR / "ckpt_v12b_d1_n96m32_tv_ratio_v14a.pt"))
    ap.add_argument("--data-file", default="")
    ap.add_argument("--data-format", default="transcripts")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--subsets", type=int, default=6, help="每块 E 采样的随机子集数")
    args = ap.parse_args()
    model, cfg = build(args.ckpt, act_x=0.2)
    _, held, vocab = load_chat(args.data_file, args.data_format)
    starts = np.linspace(0, len(held) - N_IN - N_OUT - 1, args.windows).astype(int).tolist()
    base = ce_arb(model, held, starts)
    n = model.e_n
    norms = model.memory_norms()
    ind = {}
    for i in range(n):
        ind[i] = ce_with_zero(model, held, starts, [i]) - base
    order_norm = sorted(range(n), key=lambda i: -norms[i])
    group = {}
    for k in (1, 2, 3, 5, min(10, n)):
        group[k] = ce_with_zero(model, held, starts, order_norm[:k]) - base
    # 随机子集分组重要性（大小 min(5,n)）
    size = min(5, n)
    rng = np.random.RandomState(0)
    subset_vals = {}
    for i in range(n):
        vals = []
        for _ in range(args.subsets):
            others = [j for j in range(n) if j != i]
            rng.shuffle(others)
            S = [i] + others[:size - 1]
            vals.append(ce_with_zero(model, held, starts, S) - base)
        subset_vals[i] = float(np.mean(vals))
    # M 两两余弦
    pair = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        c = float(F.cosine_similarity(model.mem[i].flatten()[None],
                                      model.mem[j].flatten()[None]))
        pair[i, j] = pair[j, i] = c
    out = {
        "ckpt": args.ckpt, "ce_base": round(base, 5),
        "norms": [round(x, 3) for x in norms],
        "individual": {str(i): round(ind[i], 5) for i in range(n)},
        "group": {str(k): round(v, 5) for k, v in group.items()},
        "subset_importance": {str(i): round(subset_vals[i], 5) for i in range(n)},
        "consolidation_order": [int(i) for i in sorted(range(n), key=lambda i: -subset_vals[i])],
        "erasure_order": [int(i) for i in sorted(range(n), key=lambda i: ind[i])],
        "pair_cos_mean": round(float(pair[np.triu_indices(n, 1)].mean()), 3),
    }
    print(f"ckpt={Path(args.ckpt).name} base_CE={base:.4f} n={n}")
    print("M norms:", [round(x, 2) for x in norms])
    print("individual:", {k: round(v, 4) for k, v in out["individual"].items()})
    print("group (top-k by norm):", out["group"])
    print("subset importance:", {k: round(v, 4) for k, v in out["subset_importance"].items()})
    print("consolidation order:", out["consolidation_order"])
    print("erasure order     :", out["erasure_order"])
    print("pair cos mean:", out["pair_cos_mean"])
    json.dump(out, open(OUT_DIR / "memory_importance.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved memory_importance.json")


if __name__ == "__main__":
    main()
