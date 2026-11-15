"""训练一个小 confidence head：用 A 解码器隐状态预测「该 token 是否可安全落地」。

标签来自 teacher-forced 验证：验证 pass 的 argmax == 块提议 token 即 label=1。
特征用正常前向的 h_a（部署时已有，不需要额外验证 pass）。

用法：
  python calibrate_conf.py --ckpt ckpt_..._zhwiki.pt \
      --data-file E:/PycharmProject/data/zhwiki_subset.txt --data-format text
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive import build
from run_online import load_chat

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent
N_IN, N_OUT = 96, 32


def collect(model, ids, act, n_iters=100, batch=4):
    feats, labels = [], []
    span = N_IN + N_OUT + 1
    for _ in range(n_iters):
        starts = np.random.randint(0, len(ids) - span - 1, size=batch)
        x = torch.stack([ids[s:s + N_IN] for s in starts]).to(DEVICE)
        with torch.no_grad():
            out0 = model(x, N_OUT, {"act": act})
            cand = out0["arb"].argmax(-1)
            out1 = model(x, N_OUT, {"act": act}, tf=cand)
            h = out0["h_a"].detach()                       # (B,m,d) 正常前向特征
            vpred = out1["verify"].argmax(-1)              # (B,m)
            lab = (vpred == cand).float().detach()         # (B,m)
        feats.append(h.reshape(-1, h.shape[-1]))
        labels.append(lab.reshape(-1))
    return torch.cat(feats, 0), torch.cat(labels, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT_DIR / "ckpt_v12b_d6_n96m32_tv_ratio_zhwiki.pt"))
    ap.add_argument("--data-file", default="E:/PycharmProject/data/zhwiki_subset.txt")
    ap.add_argument("--data-format", default="text")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--act", type=float, default=0.2)
    args = ap.parse_args()

    model, cfg = build(args.ckpt, act_x=args.act)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    train_ids, held_ids, vocab = load_chat(args.data_file, args.data_format)
    print(f"ckpt={Path(args.ckpt).name} d={model.d} act={args.act} "
          f"train={len(train_ids)} held={len(held_ids)}")

    head = nn.Linear(model.d, 1).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    feats, labels = collect(model, train_ids, args.act, n_iters=args.iters)
    print(f"train samples={len(labels)}  label rate={float(labels.mean()):.3f}")
    n = len(labels)
    bs = 4096
    for step in range(args.steps):
        idx = torch.randperm(n, device=DEVICE)[:bs]
        logit = head(feats[idx]).squeeze(-1).float()
        loss = F.binary_cross_entropy_with_logits(logit, labels[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  head s{step} bce={float(loss):.4f}")
    # held 评估
    hf, hl = collect(model, held_ids, args.act, n_iters=50)
    with torch.no_grad():
        pred = (torch.sigmoid(head(hf).squeeze(-1)) >= 0.5).float()
    acc = float((pred == hl).float().mean())
    pos = float(pred[hl > 0].mean()) if (hl > 0).any() else 0
    neg = float(pred[hl == 0].mean()) if (hl == 0).any() else 0
    print(f"held: label_rate={float(hl.mean()):.3f} acc={acc:.3f} "
          f"pred_rate_on_pos={pos:.3f} pred_rate_on_neg={neg:.3f}")
    torch.save(head.state_dict(), OUT_DIR / "conf_head.pt")
    json.dump({"ckpt": args.ckpt, "d": model.d, "label_rate_train": float(labels.mean()),
               "label_rate_held": float(hl.mean()), "held_acc": acc,
               "pred_on_pos": pos, "pred_on_neg": neg},
              open(OUT_DIR / "conf_head_stats.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved conf_head.pt / conf_head_stats.json")


if __name__ == "__main__":
    main()
