"""统计块池的角色使用：哪些块在什么时候被 A / C / M 选中（验证「不同时刻可被 A 或 C 选中」）。"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from adaptive import build
from run_online import load_chat

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT_DIR / "ckpt_v12b_d1_n96m32_tv_ratio_commit03.pt"))
    ap.add_argument("--data-file", default="")
    ap.add_argument("--data-format", default="transcripts")
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--act", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=96)
    args = ap.parse_args()
    model, cfg = build(args.ckpt, act_x=args.act)
    _, held, vocab = load_chat(args.data_file, args.data_format)
    k = cfg["k"]
    starts = np.linspace(0, len(held) - args.n - 1, args.windows).astype(int).tolist()
    role_of = [defaultdict(int) for _ in range(3)]   # role -> block -> count
    block_roles = [set() for _ in range(k)]
    quota_hist = Counter()
    with torch.no_grad():
        for s in starts:
            x = held[s:s + args.n].unsqueeze(0).to(DEVICE)
            enc = model.encode(x, args.act)
            chosen = enc["chosen"]
            quota_hist[tuple(enc["quotas"])] += 1
            for r, ids in enumerate(chosen):
                for j in ids:
                    role_of[r][j] += 1
                    block_roles[j].add(r)
    used = [j for j in range(k) if block_roles[j]]
    both = [j for j in range(k) if 0 in block_roles[j] and 2 in block_roles[j]]
    only_a = [j for j in range(k) if block_roles[j] == {0}]
    only_c = [j for j in range(k) if block_roles[j] == {2}]
    print(f"ckpt={Path(args.ckpt).name} k={k} windows={len(starts)}")
    print(f"quota 分布: {dict(quota_hist)}")
    print(f"被使用的块: {len(used)}/{k}")
    print(f"既当过 A 又当过 C 的块: {len(both)}/{k} -> {both}")
    print(f"只当过 A: {only_a}   只当过 C: {only_c}")
    for r, name in enumerate(("A", "C") if len(role_of) >= 2 else ("A",)):
        cnt = role_of[r]
        print(f"  {name} 次数 top: {sorted(cnt.items(), key=lambda kv: -kv[1])[:8]}")
    json.dump({"ckpt": args.ckpt, "k": k, "windows": len(starts),
               "used_blocks": used, "both_roles": both,
               "quota_hist": {str(k2): v for k2, v in quota_hist.items()},
               "role_counts": [dict(rc) for rc in role_of]},
              open(OUT_DIR / "role_stats.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved role_stats.json")


if __name__ == "__main__":
    main()
