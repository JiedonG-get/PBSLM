"""M → A 固化（表示/输出级蒸馏版）：把「带记忆的模型」蒸馏进 A，使去掉 M 后能力仍在。

对每块成熟 M_e：
  teacher = 完整模型（含 M）；student = 把 M_e 置零的同一模型。
  L = CE(student) + λ·KL(teacher_arb ‖ student_arb)
  对每个 A 候选单独做 k 步，实测 CE(no M) 下降 & CE(with M) 漂移，选最优保留；成功后衰减 M_e。

用法：python consolidate_repr.py --ckpt ckpt_v12b_d1_n96m32_tv_ratio_v14a.pt
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive import build
from run_online import per_token_ce, load_chat

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent
N_IN, N_OUT = 96, 32


def candidates(model):
    c = {}
    for i, layer in enumerate(model.dec_a):
        c[f"dec_a.{i}"] = list(layer.parameters())
    c["role"] = list(model.role.parameters())
    c["queries"] = [model.q_emb, model.q_pos]
    c["refine"] = list(model.refine_proj.parameters())
    return c


@contextmanager
def zero_mem(model, idx):
    saved = model.mem[idx].clone()
    with torch.no_grad():
        model.mem[idx].zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            model.mem[idx].copy_(saved)


@torch.no_grad()
def eval_ce(model, data, starts):
    ce = []
    for s in starts:
        x = data[s:s + N_IN].unsqueeze(0).to(DEVICE)
        y = data[s + N_IN:s + N_IN + N_OUT].unsqueeze(0).to(DEVICE)
        out = model(x, N_OUT, {"act": 0.2})
        ce.append(float(per_token_ce(out["arb"], y).mean()))
    return float(np.mean(ce))


def make_batches(ids, n_batch, batch, seed=0):
    rng = np.random.RandomState(seed)
    span = N_IN + N_OUT + 1
    out = []
    for _ in range(n_batch):
        starts = rng.randint(0, len(ids) - span - 1, size=batch)
        out.append((torch.stack([ids[s:s + N_IN] for s in starts]).to(DEVICE),
                    torch.stack([ids[s + N_IN:s + N_IN + N_OUT] for s in starts]).to(DEVICE)))
    return out


def dist_loss(model, x, y, e_idx, backward=True):
    """teacher 先算（含 M），student 在 M_e=0 下算，backward 必须在 restore 之前完成。"""
    t_out = model(x, N_OUT, {"act": 0.2})
    p_t = torch.softmax(t_out["arb"].float().detach(), -1).clamp_min(1e-8)
    with zero_mem(model, e_idx):
        s_out = model(x, N_OUT, {"act": 0.2})
        s_ce = per_token_ce(s_out["arb"], y).mean()
        p_s = torch.softmax(s_out["arb"].float(), -1).clamp_min(1e-8)
        kl = (p_t * (p_t.log() - p_s.log())).sum(-1).mean()
        loss = s_ce + kl
        if backward:
            loss.backward()
    return float(s_ce.detach()), float(kl.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT_DIR / "ckpt_v12b_d1_n96m32_tv_ratio_v14a.pt"))
    ap.add_argument("--data-file", default="")
    ap.add_argument("--data-format", default="transcripts")
    ap.add_argument("--mature", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mem-decay", type=float, default=0.7)
    ap.add_argument("--max-e", type=int, default=10)
    args = ap.parse_args()
    model, cfg = build(args.ckpt, act_x=0.2)
    train_ids, held, vocab = load_chat(args.data_file, args.data_format)
    cands = candidates(model)
    batches = make_batches(train_ids, args.n_batch, args.batch)
    mem_starts = np.linspace(0, len(train_ids) - N_IN - N_OUT - 1, 24).astype(int).tolist()
    held_starts = np.linspace(0, len(held) - N_IN - N_OUT - 1, 16).astype(int).tolist()
    base_with = eval_ce(model, train_ids, mem_starts)
    base_held = eval_ce(model, held, held_starts)
    print(f"ckpt={Path(args.ckpt).name} withM={base_with:.4f} held={base_held:.4f}")
    report = []
    for i in range(min(args.max_e, model.e_n)):
        if float(model.mem[i].norm()) < args.mature:
            report.append({"e": i, "skipped": True})
            continue
        res = {}
        for name, params in cands.items():
            saved = [p.detach().clone() for p in params]
            opt = torch.optim.AdamW(params, lr=args.lr)
            for step in range(args.steps):
                x, y = batches[step % len(batches)]
                model.zero_grad(set_to_none=True)
                dist_loss(model, x, y, i, backward=True)
                opt.step()
            with_m = eval_ce(model, train_ids, mem_starts)
            with zero_mem(model, i):
                no_m = eval_ce(model, train_ids, mem_starts)
            held_ce = eval_ce(model, held, held_starts)
            res[name] = (no_m - base_with, with_m - base_with, held_ce - base_held)
            with torch.no_grad():
                for p, s in zip(params, saved):
                    p.copy_(s)
        best = min(res, key=lambda k: res[k][0])
        d_noM, d_withM, d_held = res[best]
        accepted = d_noM < -1e-4 and d_withM <= 0.02
        if accepted:
            params = cands[best]
            opt = torch.optim.AdamW(params, lr=args.lr)
            for step in range(args.steps):
                x, y = batches[step % len(batches)]
                model.zero_grad(set_to_none=True)
                dist_loss(model, x, y, i, backward=True)
                opt.step()
            with torch.no_grad():
                model.mem[i].mul_(args.mem_decay)
            base_with = eval_ce(model, train_ids, mem_starts)
            base_held = eval_ce(model, held, held_starts)
        print(f"E{i:02d} -> 选 {best:9s} ΔnoM={d_noM:+.4f} ΔwithM={d_withM:+.4f} "
              f"Δheld={d_held:+.4f} {'KEEP' if accepted else 'ROLLBACK'}")
        report.append({"e": i, "chosen": best, "delta_noM": round(d_noM, 5),
                       "delta_withM": round(d_withM, 5), "delta_held": round(d_held, 5),
                       "accepted": bool(accepted),
                       "all": {k: [round(v[0], 5), round(v[1], 5), round(v[2], 5)]
                               for k, v in res.items()}})
    json.dump(report, open(OUT_DIR / "consolidate_repr_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved consolidate_repr_results.json")


if __name__ == "__main__":
    main()
