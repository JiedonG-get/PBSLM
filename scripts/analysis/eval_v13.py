import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

"""v13 补充量化：流动一致性 I 的分布、活跃度上限 x% 带来的延迟收益。"""
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.model import PBSLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent
CHAT = Path("E:/PycharmProject/transcripts")
CKPT = OUT_DIR / "ckpt_v12b_d6_n96m32_tv_ratio_v13long.pt"
N_IN, N_OUT = 96, 32
N_WIN = 16


def load_data():
    files = sorted(CHAT.glob("*.md"))
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(len(text) * 0.85)
    return ids[:cut], ids[cut:], vocab


def build(act_x, state):
    ck = torch.load(CKPT, weights_only=False, map_location=DEVICE)
    cfg = ck["config"]
    model = PBSLM(cfg["vocab"] if "vocab" in cfg else ck["state"]["tok.weight"].shape[0],
                  k=cfg["k"], h=cfg["h"], depth=cfg["depth"],
                  m_max=cfg.get("m_max", 128), arb=cfg.get("arb", "consistency"),
                  refine=cfg.get("refine", 2),
                  self_causal=not cfg.get("no_self_causal", False),
                  flow_dim=cfg.get("flow_dim", 64), flow_steps=cfg.get("flow_steps", 2),
                  act_x=act_x).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def ce(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()


def main():
    ck = torch.load(CKPT, weights_only=False, map_location=DEVICE)
    state, cfg = ck["state"], ck["config"]
    _, held, vocab = load_data()
    model = build(cfg.get("act_x", 0.2), state)
    starts = np.linspace(0, len(held) - N_IN - N_OUT - 1, N_WIN).astype(int).tolist()
    rows = []
    with torch.no_grad():
        for s in starts:
            x = held[s:s + N_IN].unsqueeze(0).to(DEVICE)
            y = held[s + N_IN:s + N_IN + N_OUT].unsqueeze(0).to(DEVICE)
            out = model(x, N_OUT, {"act": 0.2})
            rows.append({
                "arb": ce(out["arb"], y), "a": ce(out["a"], y), "c": ce(out["c"], y),
                "Ia": float(out["cons_a"].mean()), "Ic": float(out["cons_c"].mean()),
                "g": float(out["g"].mean()), "active": out["aux"]["active"],
            })
    summ = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
    summ["Ia_std"] = round(float(np.std([r["Ia"] for r in rows])), 4)
    summ["Ic_std"] = round(float(np.std([r["Ic"] for r in rows])), 4)
    summ["g_std"] = round(float(np.std([r["g"] for r in rows])), 4)
    print(f"I_a={summ['Ia']:.3f}±{summ['Ia_std']:.3f}  I_c={summ['Ic']:.3f}±{summ['Ic_std']:.3f} "
          f"g={summ['g']:.3f}±{summ['g_std']:.3f}  active={summ['active']:.0f}")
    print(f"held CE: arb={summ['arb']:.3f} a={summ['a']:.3f} c={summ['c']:.3f}")

    # ---- 延迟：同一套权重，act_x（上限）不同 -> 启用块数不同
    lat = {}
    x = held[:N_IN].unsqueeze(0).to(DEVICE)
    for cap, act in ((0.2, 0.2), (0.2, 0.1), (1.0, 1.0)):
        m = build(cap, state)
        with torch.no_grad():
            for _ in range(2):
                m(x, N_OUT, {"act": act})
            torch.cuda.synchronize()
            st, en = torch.cuda.Event(True), torch.cuda.Event(True)
            st.record()
            for _ in range(5):
                m(x, N_OUT, {"act": act})
            en.record()
            torch.cuda.synchronize()
        ms = st.elapsed_time(en) / 5
        active = m(x, N_OUT, {"act": act})["aux"]["active"] if False else None
        # 重新取一次 active
        with torch.no_grad():
            act_n = m(x, N_OUT, {"act": act})["aux"]["active"]
        lat[f"cap={cap},act={act}"] = {"active_blocks": act_n, "fwd_ms": round(ms, 2)}
        print(f"cap={cap} act={act}: active={act_n} blocks, fwd={ms:.2f} ms")
    json.dump({"ckpt": str(CKPT), "eval": summ, "latency": lat},
              open(OUT_DIR / "v13_flow_latency.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved v13_flow_latency.json")


if __name__ == "__main__":
    main()
