"""对比：信息一致性候选 / 冲击评分候选 / 在线稳定性范围 / 更长预训练 + depth3。"""
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from model import PBSLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.dirname(os.path.abspath(__file__))
CHAT_DIR = "E:/PycharmProject/transcripts"


def load_data():
    files = sorted(glob.glob(os.path.join(CHAT_DIR, "*.md")))
    text = "\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in files)
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(len(text) * 0.85)
    return ids[:cut], ids[cut:]


def per_token_ce(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                           reduction="none").view_as(y)


@torch.no_grad()
def eval_pair(model, data, n, m, n_win=16):
    starts = np.linspace(0, len(data) - n - m - 1, n_win).astype(int).tolist()
    ce = {"a": [], "c": [], "arb": []}
    gs = []
    for s in starts:
        x = data[s:s + n].unsqueeze(0).to(DEVICE)
        y = data[s + n:s + n + m].unsqueeze(0).to(DEVICE)
        out = model(x, m, None)
        for k in ("a", "c", "arb"):
            ce[k].append(float(per_token_ce(out[k], y).mean()))
        gs.append(float(out["g"].mean()))
    return {"arb": round(float(np.mean(ce["arb"])), 4),
            "a": round(float(np.mean(ce["a"])), 4),
            "c": round(float(np.mean(ce["c"])), 4),
            "g_mean": round(float(np.mean(gs)), 4)}, len(starts)


def load_ckpt(path):
    ck = torch.load(path, weights_only=False, map_location=DEVICE)
    cfg = ck["config"]
    vocab = ck["state"]["tok.weight"].shape[0]
    model = PBSLM(vocab, k=cfg["k"], h=cfg["h"], depth=cfg["depth"],
                  m_max=cfg.get("m_max", 128), arb=cfg.get("arb", "consistency"),
                  refine=cfg.get("refine", 2),
                  self_causal=not cfg.get("no_self_causal", False),
                  cons_metric=cfg.get("cons", "tv")).to(DEVICE)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, cfg


def main():
    _, held = load_data()
    out = {"consistency": [], "impact": [], "stability": [], "depth": []}

    # --- 信息一致性候选（chunks=0 预训练 800 步的 ckpt）
    for p in sorted(glob.glob(os.path.join(OUT, "ckpt_v12b_d1_n96m32_*_w1*.pt"))):
        model, cfg = load_ckpt(p)
        r, nw = eval_pair(model, held, 96, 32)
        row = {"ckpt": os.path.basename(p), "cons": cfg.get("cons"),
               "impact": cfg.get("impact"), **r, "windows": nw}
        out["consistency"].append(row)
        print(f"[cons] {row['cons']:5s} impact={row['impact']:7s} held arb={r['arb']:.3f} "
              f"a={r['a']:.3f} c={r['c']:.3f} g={r['g_mean']:.3f}")

    # --- 冲击评分候选（cons=tv, 800 步）
    for p in sorted(glob.glob(os.path.join(OUT, "ckpt_v12b_d1_n96m32_tv_*_w2*.pt"))):
        model, cfg = load_ckpt(p)
        r, nw = eval_pair(model, held, 96, 32)
        row = {"ckpt": os.path.basename(p), "cons": cfg.get("cons"),
               "impact": cfg.get("impact"), **r, "windows": nw}
        out["impact"].append(row)
        print(f"[impact] {row['impact']:7s} held arb={r['arb']:.3f} a={r['a']:.3f} "
              f"c={r['c']:.3f} g={r['g_mean']:.3f}")

    # --- 精修轮次 / query 因果自注意力（chunks=0 的 ckpt）
    for p in sorted(glob.glob(os.path.join(OUT, "ckpt_v12b_d1_n96m32_*_w5*.pt"))):
        model, cfg = load_ckpt(p)
        r, nw = eval_pair(model, held, 96, 32)
        row = {"ckpt": os.path.basename(p), "refine": cfg.get("refine", 2),
               "self_causal": not cfg.get("no_self_causal", False),
               **r, "windows": nw}
        out.setdefault("refine", []).append(row)
        print(f"[refine] rounds={row['refine']} self_causal={row['self_causal']} "
              f"held arb={r['arb']:.3f} a={r['a']:.3f} c={r['c']:.3f} g={r['g_mean']:.3f}")

    # --- 更长预训练 / depth3（从结果 JSON 读在线前后）
    for p in sorted(glob.glob(os.path.join(OUT, "results_v12_d*_w4*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        cfg = r["config"]
        row = {"file": os.path.basename(p), "depth": cfg["depth"], "steps": cfg.get("steps"),
               "params": r["model_params"]["total"],
               "held_before": r["held_before"]["main"],
               "held_after": r["held_after"],
               "sweep": r["sweep_after"]}
        out["depth"].append(row)
        print(f"[depth] d{row['depth']} steps={row['steps']} params={row['params']/1e6:.2f}M "
              f"held {row['held_before']['arb']:.3f} -> {row['held_after']['arb']:.3f}")

    # --- 在线稳定性（从结果 JSON 读 records）
    for p in sorted(glob.glob(os.path.join(OUT, "results_v12_d*_w3*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        cfg = r["config"]
        rec = r["online"]["records"]
        g = np.array(rec.get("g", [0.0]))
        row = {"file": os.path.basename(p), "scope": cfg.get("online_scope"),
               "replay": cfg.get("replay"), "lr": cfg.get("online_lr"),
               "held_before": r["held_before"]["main"], "held_after": r["held_after"],
               "stream_first500": float(np.mean(rec["ce_arb"][:500])),
               "stream_last500": float(np.mean(rec["ce_arb"][-500:])),
               "g_mean": float(g.mean()) if len(g) else None,
               "g_std": float(g.std()) if len(g) else None,
               "held_curve": [(h["chunk"], h["held"]["arb"]) for h in r["online"]["held_curve"]]}
        out["stability"].append(row)
        print(f"[stability] scope={row['scope']:10s} replay={row['replay']} "
              f"held {row['held_before']['arb']:.3f} -> {row['held_after']['arb']:.3f} "
              f"stream {row['stream_first500']:.3f}->{row['stream_last500']:.3f} "
              f"g={row['g_mean']}±{row['g_std']}")

    json.dump(out, open(os.path.join(OUT, "metrics_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved metrics_summary.json")


if __name__ == "__main__":
    main()
