"""系统长度外推评测：用已训练 checkpoint，扫 n（上下文长度）× m（一次输出 token 数）。

不重新训练，直接加载 ckpt_v12b_d1_n96m32_lr3e4.pt（训练时只见过 n=96→m=32）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import PBSLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent
CHAT_DIR = Path("E:/PycharmProject/transcripts")
CKPT = OUT_DIR / "ckpt_v12b_d1_n96m32_lr3e4.pt"

N_LIST = [16, 32, 64, 96, 128, 192, 256, 384, 512]
M_LIST = [4, 8, 16, 32, 64, 128]
N_WIN = 32


def load_data():
    files = sorted(CHAT_DIR.glob("*.md"))
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(len(text) * 0.85)
    return ids[:cut], ids[cut:], vocab


def per_token_ce(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                           reduction="none").view_as(y)


@torch.no_grad()
def eval_pair(model, data, n, m, n_win=N_WIN):
    starts = np.linspace(0, len(data) - n - m - 1, min(n_win, max(1, len(data) // (n + m) - 1)))
    starts = starts.astype(int).tolist()
    ce = {"a": [], "c": [], "arb": []}
    top1 = []
    for s in starts:
        x = data[s:s + n].unsqueeze(0).to(DEVICE)
        y = data[s + n:s + n + m].unsqueeze(0).to(DEVICE)
        out = model(x, m, None)
        for k in ("a", "c", "arb"):
            ce[k].append(float(per_token_ce(out[k], y).mean()))
        pred = out["arb"].argmax(-1)
        top1.append(float((pred == y).float().mean()))
    return {"n_in": n, "n_out": m, "windows": len(starts),
            "arb": round(float(np.mean(ce["arb"])), 4),
            "a": round(float(np.mean(ce["a"])), 4),
            "c": round(float(np.mean(ce["c"])), 4),
            "top1": round(float(np.mean(top1)), 4),
            "arb_std": round(float(np.std(ce["arb"])), 4)}


def main():
    ckpt = torch.load(CKPT, weights_only=False, map_location=DEVICE)
    cfg = ckpt["config"]
    state = ckpt["state"]
    vocab = state["tok.weight"].shape[0]
    model = PBSLM(vocab, k=cfg["k"], h=cfg["h"], depth=cfg["depth"],
                  m_max=cfg.get("m_max", 128), arb=cfg.get("arb", "consistency"),
                  refine=cfg.get("refine", 2),
                  self_causal=not cfg.get("no_self_causal", False)).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    _, held, vocab2 = load_data()
    assert len(vocab2) == vocab, (len(vocab2), vocab)
    print(f"ckpt={CKPT.name} depth={cfg['depth']} refine={cfg.get('refine')} "
          f"trained n={cfg['n_in']} m={cfg['n_out']}")
    rows = []
    for n in N_LIST:
        line = []
        for m in M_LIST:
            if n + m + 1 >= len(held) - 2:
                continue
            r = eval_pair(model, held, n, m)
            rows.append(r)
            line.append(f"{r['arb']:.3f}")
        print(f"n={n:4d}: " + "  ".join(line))
    print("m list:", M_LIST)
    json.dump({"ckpt": str(CKPT), "trained": {"n": cfg["n_in"], "m": cfg["n_out"]},
               "n_list": N_LIST, "m_list": M_LIST, "windows": N_WIN, "rows": rows},
              open(OUT_DIR / "length_extrapolation.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved length_extrapolation.json")


if __name__ == "__main__":
    main()
