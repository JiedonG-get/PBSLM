"""M → A 固化 v2：用「实测」选最适合这块 M 的 A 组件（counterfactual selection）。

对每个成熟 M_e：
  1) 在 replay 批次上，对每个 A 候选组件单独做 k 步 L_cons = CE(A) + λ·MSE(A 中间表示, M_e 读出)；
  2) 实测该候选带来的 replay CE 下降（held CE 漂移作护栏）；
  3) 选下降最大的候选并保留其参数；若没有下降/held 变差则完全回滚；
  4) 成功后衰减 M_e（记忆 → 能力，释放容量）。

用法：python consolidate.py --ckpt ckpt_v12b_d1_n96m32_tv_ratio_v14a.pt
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


def candidates(model):
    c = {}
    for i, layer in enumerate(model.dec_a):
        c[f"dec_a.{i}"] = list(layer.parameters())
    c["role"] = list(model.role.parameters())
    c["queries"] = [model.q_emb, model.q_pos]
    c["refine"] = list(model.refine_proj.parameters())
    return c


@torch.no_grad()
def eval_ce(model, data, starts, n_win=None):
    model.eval()
    ce = []
    starts = starts if n_win is None else starts
    for s in starts:
        x = data[s:s + N_IN].unsqueeze(0).to(DEVICE)
        y = data[s + N_IN:s + N_IN + N_OUT].unsqueeze(0).to(DEVICE)
        out = model(x, N_OUT, {"act": 0.2})
        ce.append(float(per_token_ce(out["arb"], y).mean()))
    return float(np.mean(ce))


def cons_loss(model, x, y, e_idx):
    out = model(x, N_OUT, {"act": 0.2})
    ce = per_token_ce(out["arb"], y).mean()
    e = out["e_feats"][e_idx]
    r = torch.einsum("btd,dv->btv", e, model.mem[e_idx])[:, -N_OUT:]
    dist = F.mse_loss(model.a_pred(out["h_a"]), r.detach())
    return ce + 0.5 * dist, ce


def make_batches(ids, n_batch, batch, seed=0):
    rng = np.random.RandomState(seed)
    span = N_IN + N_OUT + 1
    out = []
    for _ in range(n_batch):
        starts = rng.randint(0, len(ids) - span - 1, size=batch)
        out.append((torch.stack([ids[s:s + N_IN] for s in starts]).to(DEVICE),
                    torch.stack([ids[s + N_IN:s + N_IN + N_OUT] for s in starts]).to(DEVICE)))
    return out


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
    base_mem = eval_ce(model, train_ids, mem_starts)
    base_held = eval_ce(model, held, held_starts)
    print(f"ckpt={Path(args.ckpt).name} replay_CE={base_mem:.4f} held_CE={base_held:.4f} E={model.e_n}")
    report = []
    for i in range(min(args.max_e, model.e_n)):
        mnorm = float(model.mem[i].norm())
        if mnorm < args.mature:
            report.append({"e": i, "m_norm": mnorm, "skipped": True})
            continue
        results = {}
        for name, params in cands.items():
            saved = [p.detach().clone() for p in params]
            opt = torch.optim.AdamW(params, lr=args.lr)
            for step in range(args.steps):
                x, y = batches[step % len(batches)]
                model.zero_grad(set_to_none=True)
                loss, _ = cons_loss(model, x, y, i)
                loss.backward()
                opt.step()
            dm = eval_ce(model, train_ids, mem_starts) - base_mem
            dh = eval_ce(model, held, held_starts) - base_held
            results[name] = (dm, dh)
            with torch.no_grad():
                for p, s in zip(params, saved):
                    p.copy_(s)
        best = min(results, key=lambda k: results[k][0])
        dm, dh = results[best]
        accepted = dm < -1e-4 and dh <= 0.02
        if accepted:
            # 重做最佳候选的加训并保留
            params = cands[best]
            opt = torch.optim.AdamW(params, lr=args.lr)
            for step in range(args.steps):
                x, y = batches[step % len(batches)]
                model.zero_grad(set_to_none=True)
                loss, _ = cons_loss(model, x, y, i)
                loss.backward()
                opt.step()
            with torch.no_grad():
                model.mem[i].mul_(args.mem_decay)
            base_mem = eval_ce(model, train_ids, mem_starts)
            base_held = eval_ce(model, held, held_starts)
        print(f"E{i:02d} M_norm={mnorm:.2f} -> 选 {best:9s} "
              f"Δreplay={dm:+.4f} Δheld={dh:+.4f} "
              f"{'KEEP' if accepted else 'ROLLBACK'}")
        report.append({"e": i, "m_norm": mnorm, "chosen": best,
                       "delta_mem": round(dm, 5), "delta_held": round(dh, 5),
                       "accepted": bool(accepted),
                       "all": {k: [round(v[0], 5), round(v[1], 5)] for k, v in results.items()}})
    json.dump(report, open(OUT_DIR / "consolidate_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved consolidate_results.json")


if __name__ == "__main__":
    main()
