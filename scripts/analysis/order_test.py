"""记忆擦除顺序 vs 固化顺序对照（用最佳 v14/C-isolated 模型）。

擦除：按单块重要性升序（最冗余先擦）/ 降序 / 随机，擦 3 块，记录 CE 轨迹。
固化：按分组重要性（单块近似）排序，逐块做 2 步 teacher→student 蒸馏（M_e 置零），
      记录 CE(with M) 与 CE(no M) 轨迹；比较不同顺序的最终效果。
"""
import argparse
import json
import random
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
def ce(model, data, starts):
    vals = []
    for s in starts:
        x = data[s:s + N_IN].unsqueeze(0).to(DEVICE)
        y = data[s + N_IN:s + N_IN + N_OUT].unsqueeze(0).to(DEVICE)
        out = model(x, N_OUT, {"act": 0.2})
        vals.append(float(per_token_ce(out["arb"], y).mean()))
    return float(np.mean(vals))


@torch.no_grad()
def ce_zero(model, data, starts, idxs):
    saved = [model.mem[i].clone() for i in idxs]
    for i in idxs:
        model.mem[i].zero_()
    try:
        return ce(model, data, starts)
    finally:
        for i, sv in zip(idxs, saved):
            model.mem[i].copy_(sv)


def distill_step(model, x, y, e_idx, lr=1e-5):
    A = model.online_params()
    saved = [p.detach().clone() for p in A]
    opt = torch.optim.AdamW(A, lr=lr)
    model.zero_grad(set_to_none=True)
    t_out = model(x, N_OUT, None)
    p_t = torch.softmax(t_out["arb"].float().detach(), -1).clamp_min(1e-8)
    saved_m = model.mem[e_idx].clone()
    model.mem[e_idx].zero_()
    s_out = model(x, N_OUT, None)
    s_ce = per_token_ce(s_out["arb"].float(), y).mean()
    p_s = torch.softmax(s_out["arb"].float(), -1).clamp_min(1e-8)
    loss = s_ce + (p_t * (p_t.log() - p_s.log())).sum(-1).mean()
    loss.backward()
    model.mem[e_idx].copy_(saved_m)
    opt.step()
    return [p.detach().clone() for p in A], saved  # new, old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT_DIR / "ckpt_v12b_d6_n96m32_tv_ratio_v15_best.pt"))
    ap.add_argument("--data-file", default="E:/PycharmProject/data/zhwiki_subset.txt")
    ap.add_argument("--data-format", default="text")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()
    model, cfg = build(args.ckpt, act_x=0.2)
    train_ids, held, vocab = load_chat(args.data_file, args.data_format)
    starts = np.linspace(0, len(held) - N_IN - N_OUT - 1, args.windows).astype(int).tolist()
    base = ce(model, held, starts)
    n = model.e_n
    ind = {i: ce_zero(model, held, starts, [i]) - base for i in range(n)}
    rank_asc = sorted(range(n), key=lambda i: ind[i])        # 最冗余先
    rank_desc = list(reversed(rank_asc))
    rng = random.Random(0)
    rank_rand = list(range(n)); rng.shuffle(rank_rand)
    print(f"ckpt={Path(args.ckpt).name} base={base:.4f}")
    print("individual importance:", {i: round(ind[i], 4) for i in range(n)})
    out = {"ckpt": args.ckpt, "base": base,
           "individual": {str(i): round(ind[i], 5) for i in range(n)},
           "erase_orders": {"asc_redundant": rank_asc[:3], "desc_important": rank_desc[:3],
                            "random": rank_rand[:3]}}
    # 擦除顺序
    erase = {}
    for name, order in (("asc_redundant", rank_asc), ("desc_important", rank_desc),
                        ("random", rank_rand)):
        model.mem_saved = None
        saved = [m.clone() for m in model.mem]
        traj = [base]
        for i in order[:3]:
            with torch.no_grad():
                model.mem[i].zero_()
            traj.append(ce(model, held, starts))
        erase[name] = [round(v, 5) for v in traj]
        for m, sv in zip(model.mem, saved):
            with torch.no_grad():
                m.copy_(sv)
        print(f"erase {name:14s} order={order[:3]} CE轨迹={erase[name]}")
    out["erase_traj"] = erase
    # 固化顺序（用单块重要性排序；逐块 2 步蒸馏）
    batch = None
    starts_b = np.random.RandomState(0).randint(0, len(train_ids) - N_IN - N_OUT - 1, size=2)
    xs = torch.stack([train_ids[s:s + N_IN] for s in starts_b]).to(DEVICE)
    ys = torch.stack([train_ids[s + N_IN:s + N_IN + N_OUT] for s in starts_b]).to(DEVICE)
    cons = {}
    for name, order in (("important_first", rank_desc), ("redundant_first", rank_asc),
                        ("random", rank_rand)):
        A = model.online_params()
        saved_A = [p.detach().clone() for p in A]
        traj = []
        for i in order[:3]:
            for _ in range(args.steps):
                distill_step(model, xs, ys, i)
            traj.append([round(ce(model, held, starts), 5),
                         round(ce_zero(model, held, starts, [i]), 5)])
        cons[name] = traj
        with torch.no_grad():
            for p, sv in zip(A, saved_A):
                p.copy_(sv)
        print(f"consolidate {name:15s} order={order[:3]} (withM,noM)轨迹={traj}")
    out["consolidate_traj"] = cons
    json.dump(out, open(OUT_DIR / "order_test.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved order_test.json")


if __name__ == "__main__":
    main()
