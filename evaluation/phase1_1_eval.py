"""Phase 1.1 评估：固定 held-out 集 + 上下文敏感性/效用/泛化 + 自然消融。

不训练、不改模型；只加载已有 checkpoint 做低成本评估。
指标定义见 docs/EVAL_PROTOCOL.md。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_v21 import build_model, load_ids  # noqa: E402


def build_windows(data, n_in, ar_m, windows):
    starts = np.linspace(0, len(data) - n_in - ar_m - 2, windows).astype(int)
    xs, ys = [], []
    for s in starts:
        xs.append(np.stack([data[s + j:s + j + n_in] for j in range(ar_m)]))
        ys.append([data[s + n_in + j] for j in range(ar_m)])
    return np.stack(xs), np.stack(ys)


def predict(model, x, batch_size, code, device):
    """x: (N,n_in) long -> preds, top5_preds, z(float32 cpu)."""
    preds, top5s, zs = [], [], []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].to(device).long()
            z = model(xb).float()                       # (B,d)
            sim = z @ code.t()                           # (B,V)
            preds.append(sim.argmax(-1).cpu())
            k = sim.topk(5, dim=-1).indices
            top5s.append(k.cpu())
            zs.append(z.cpu())
    return torch.cat(preds), torch.cat(top5s), torch.cat(zs)


def freq_report(y, preds, top5s, freq, thresholds):
    out = {}
    y_np = y.cpu().numpy()
    p_np = preds.cpu().numpy()
    t_np = top5s.cpu().numpy()
    edges = thresholds  # [0.01, 0.001, 1e-4]
    for name, lo, hi in (
        ("high", edges[0], 1.0),
        ("mid", edges[1], edges[0]),
        ("low", edges[2], edges[1]),
        ("rare", 0.0, edges[2]),
    ):
        mask = (freq[y_np] >= lo) & (freq[y_np] < hi)
        n = int(mask.sum())
        if n == 0:
            out[name] = {"n": 0, "top1": None, "top5": None}
            continue
        yy = y_np[mask]
        out[name] = {
            "n": n,
            "top1": float((p_np[mask] == yy).mean()),
            "top5": float((t_np[mask] == yy[:, None]).any(-1).mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/eval_phase1_1.json")
    ap.add_argument("--tag", default="a1")
    ap.add_argument("--windows", type=int, default=-1, help="覆盖 eval.windows")
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        ecfg = json.load(f)
    ev = ecfg["eval"]
    data_cfg = ecfg["data"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    windows = args.windows if args.windows > 0 else ev["windows"]
    batch = ev["batch"]
    seed = ev["seed"]
    n_in, ar_m = data_cfg["n_in"], data_cfg["ar_m"]

    ck = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    cfg = ck["config"]
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    code = F.normalize(model.code_table.weight.float().to(device), dim=-1)
    train_data, held_data = load_ids(data_cfg["ids"], "uint32", data_cfg["hold"])
    freq = np.bincount(train_data, minlength=cfg["model"]["vocab"]).astype(np.float64)
    freq /= freq.sum()

    t0 = time.time()
    xw, yw = build_windows(held_data, n_in, ar_m, windows)
    x = torch.from_numpy(xw.reshape(-1, n_in).copy()).long()
    y = torch.from_numpy(yw.reshape(-1).copy()).long()
    N = len(y)
    print(f"targets={N} windows={windows} variants=6 device={device}", flush=True)

    # 原始上下文
    pred, top5, zorig = predict(model, x, batch, code, device)
    ydev = y.to(device)
    p1 = pred.cpu(); p5 = top5.cpu()
    orig_acc = float((p1 == y).float().mean())
    orig_t5 = float((p5 == y[:, None]).any(-1).float().mean())
    print(f"original top1={orig_acc:.4f} top5={orig_t5:.4f} sec={time.time()-t0:.1f}", flush=True)

    # 固定的自然上下文替换（target 不变）
    rng = np.random.default_rng(seed + 101)
    other_starts = rng.integers(0, len(held_data) - n_in, size=N)
    x_other = torch.from_numpy(
        np.stack([held_data[s:s + n_in].copy() for s in other_starts])).long()

    variants = {}
    variants["matched_natural_context"] = x_other
    variants["last_token_only"] = x[:, -1:].expand(-1, n_in).clone()
    xv = x.clone(); xv[:, :n_in // 2] = ev["unk_id"]; variants["drop_prefix"] = xv
    xv = x.clone(); k = max(1, n_in // 4); xv[:, -k:] = ev["unk_id"]; variants["drop_suffix"] = xv
    g = torch.Generator().manual_seed(seed + 202)
    perm = torch.argsort(torch.rand(N, n_in - 1, generator=g), dim=1)
    xv = x.clone(); xv[:, :-1] = x[:, :-1].gather(1, perm); variants["prefix_shuffle"] = xv

    result = {}
    for name, xv in variants.items():
        pv, t5v, zv = predict(model, xv, batch, code, device)
        acc = float((pv == y).float().mean())
        acc5 = float((t5v == y[:, None]).any(-1).float().mean())
        change = float((pv != p1).float().mean())
        cos_shift = float((1 - F.cosine_similarity(zv.float(), zorig.float(), dim=-1)).mean())
        result[name] = {
            "top1": acc, "top5": acc5,
            "pred_change_rate_vs_orig": change,
            "cos_shift_vs_orig": cos_shift,
            "top1_delta_vs_orig": acc - orig_acc,
            "top5_delta_vs_orig": acc5 - orig_t5,
        }
        print(f"{name}: top1={acc:.4f} change={change:.4f} cos_shift={cos_shift:.4f} "
              f"sec={time.time()-t0:.1f}", flush=True)

    out = {
        "checkpoint": str(args.checkpoint), "step": int(ck.get("step", -1)),
        "codebook": ck.get("codebook_mode"), "device": str(device),
        "eval_windows": windows, "n_targets": N,
        "data": {"ids": data_cfg["ids"], "n_in": n_in, "ar_m": ar_m, "hold": data_cfg["hold"]},
        "context_generalization": {
            "top1": orig_acc, "top5": orig_t5,
            "freq_bucket": freq_report(y, p1, p5, freq, ev["freq_thresholds"]),
        },
        "context_sensitivity": {
            "definition": "fixed target, changed context; matched natural context",
            "matched_natural_context": result["matched_natural_context"],
        },
        "context_utility": {
            "definition": "fixed target, natural/controlled context ablations; positive delta = original context helps",
            **{k: v for k, v in result.items() if k != "matched_natural_context"},
        },
        "definitions": {
            "freq_thresholds": ev["freq_thresholds"],
            "unk_id": ev["unk_id"],
            "last_token_only": "repeat last context token across all n_in positions",
            "drop_prefix": f"first {n_in // 2} context tokens -> unk",
            "drop_suffix": f"last {max(1, n_in // 4)} context tokens -> unk",
            "prefix_shuffle": "shuffle first n_in-1 tokens; last token unchanged",
            "matched_natural_context": "natural held context from another window; original target kept",
        },
        "cost": {"seconds": round(time.time() - t0, 1), "forward_variants": len(variants) + 1},
    }
    outdir = Path(ecfg["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"eval_{args.tag}.json"
    outpath.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {outpath} total_sec={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
