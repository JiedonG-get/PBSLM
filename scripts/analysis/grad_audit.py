"""梯度连通性审计：单步 backward，检查每个参数组是否有梯度。

用法：python grad_audit.py --ckpt ckpt_v12b_d6_n96m32_tv_ratio_long_p0.pt
"""
import argparse
from collections import defaultdict

import numpy as np
import torch

from adaptive import build
from run_online import per_token_ce, shock_score

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IDS = "E:/PycharmProject/data/zhwiki_full_u16.bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_v12b_d6_n96m32_tv_ratio_long_p0.pt")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    ids = np.memmap(IDS, dtype=np.uint16, mode="r")
    cut = int(len(ids) * 0.85)
    train = np.asarray(ids[:cut])
    model, cfg = build(args.ckpt, act_x=0.2)
    model.train()
    st = torch.randint(0, len(train) - 129, (args.batch,))
    x = torch.stack([torch.from_numpy(train[s:s + 96].copy()) for s in st]).to(DEV).long()
    y = torch.stack([torch.from_numpy(train[s + 96:s + 128].copy()) for s in st]).to(DEV).long()

    def loss_fn():
        out = model(x, 32, {"act": 0.2}, tf=y)
        ce_a = per_token_ce(out["a"].float(), y)
        ce_c = per_token_ce(out["c"].float(), y)
        ce_arb = per_token_ce(out["arb"].float(), y)
        S, w = shock_score(out["a"], out["c"], y, 0.5)
        rep = ((model.a_pred(out["h_a"]) - model.tok(y).detach()) ** 2).mean(-1)
        loss = ce_arb.mean() + 0.5 * (ce_a * w).mean() + 0.5 * ce_c.mean() + 0.1 * (rep * w).mean()
        p_blk = torch.softmax(out["arb"].float(), -1).clamp_min(1e-8)
        p_ver = torch.softmax(out["verify"].float().detach(), -1).clamp_min(1e-8)
        p_ver = p_ver / p_ver.sum(-1, keepdim=True)
        kl = (p_blk * (p_blk.log() - p_ver.log())).sum(-1).mean()
        return loss + 0.3 * kl, out["aux"]["chosen"]

    model.zero_grad(set_to_none=True)
    loss, chosen = loss_fn()
    loss.backward()
    print("loss", round(float(loss), 3), "chosen A/C", chosen)
    groups = defaultdict(lambda: [0, 0, 0.0])
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = ".".join(name.split(".")[:2])
        groups[key][0] += 1
        if p.grad is not None:
            groups[key][1] += 1
            groups[key][2] = max(groups[key][2], float(p.grad.norm()))
    print(f"{'group':24s} params  with_grad  max_grad_norm")
    dead = []
    for k in sorted(groups):
        tot, have, mx = groups[k]
        flag = "" if have == tot else "  <-- DEAD"
        if have != tot:
            dead.append(k)
        print(f"{k:24s} {tot:6d} {have:9d}  {mx:.3e}{flag}")
    none = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    print("\nparams with grad=None:", none if none else "none")
    print("dead groups:", dead if dead else "none")


if __name__ == "__main__":
    main()
