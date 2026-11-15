"""汇总 v12b 各分支：仲裁恢复 + 迭代精修 + 在线稳定性 / 深度。"""
import glob
import json
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "results_v12_d*.json"))):
        base = os.path.basename(p)
        if "_smoke" in base or base in ("results_v12_d1_n96m32.json", "results_v12_d2_n96m32.json"):
            continue
        r = json.load(open(p, encoding="utf-8"))
        cfg = r["config"]
        rec = r["online"]["records"]
        n = len(rec["ce_arb"])
        g = np.array(rec.get("g", [0.0]))

        def mean_last(x, w=100):
            return float(np.mean(x[-w:])) if len(x) else 0.0

        rows.append({
            "file": os.path.basename(p),
            "depth": cfg["depth"], "arb": cfg["arb"], "refine": cfg["refine"],
            "self_causal": not cfg["no_self_causal"],
            "online_lr": cfg["online_lr"], "online_decay": cfg["online_decay"],
            "params": r["model_params"], "pretrain_seconds": r["pretrain"]["seconds"],
            "loss_first": r["pretrain"]["loss_curve"][0],
            "loss_last": r["pretrain"]["loss_curve"][-1],
            "held_before": r["held_before"]["main"],
            "held_after": r["held_after"],
            "d_arb": round(r["held_after"]["arb"] - r["held_before"]["main"]["arb"], 4),
            "stream_first100": float(np.mean(rec["ce_arb"][:100])),
            "stream_last100": mean_last(rec["ce_arb"]),
            "stream_first500": float(np.mean(rec["ce_arb"][:500])),
            "stream_last500": mean_last(rec["ce_arb"], 500),
            "g_mean": float(g.mean()), "g_std": float(g.std()),
            "g_last100_mean": mean_last(g), "g_last100_std": float(np.std(g[-100:])),
            "held_curve": [(h["chunk"], h["held"]["arb"], h["held"]["a"])
                           for h in r["online"]["held_curve"]],
            "sweep_before": r["held_before"]["sweep"],
            "sweep_after": r["sweep_after"],
            "checkpoint": r.get("checkpoint"),
        })
    return rows


def main():
    rows = load()
    json.dump(rows, open(os.path.join(OUT, "v12b_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"{'arm':26s} {'depth':>5s} {'lr':>6s} {'decay':>5s} {'params':>8s} "
          f"{'held_before':>11s} {'held_after':>10s} {'d':>7s} {'stream1st':>9s} {'streamLast':>10s} "
          f"{'g_mean':>6s} {'g_std':>6s}")
    for r in rows:
        name = r["file"].replace("results_v12b_", "").replace(".json", "")
        print(f"{name:26s} {r['depth']:5d} {r['online_lr']:6.0e} {str(r['online_decay']):>5s} "
              f"{r['params']['total']/1e6:7.2f}M {r['held_before']['arb']:11.3f} "
              f"{r['held_after']['arb']:10.3f} {r['d_arb']:+7.3f} "
              f"{r['stream_first500']:9.3f} {r['stream_last500']:10.3f} "
              f"{r['g_mean']:6.3f} {r['g_std']:6.3f}")
    print("\n=== n/m sweep (held arb CE, after online) ===")
    pairs = [(s["n_in"], s["n_out"]) for s in rows[0]["sweep_after"]]
    print(f"{'arm':26s} " + " ".join(f"{n:>3d}->{m:<3d}" for n, m in pairs))
    for r in rows:
        vals = {s["n_in"] * 1000 + s["n_out"]: s["arb"] for s in r["sweep_after"]}
        name = r["file"].replace("results_v12b_", "").replace(".json", "")
        print(f"{name:26s} " + " ".join(f"{vals.get(n*1000+m, float('nan')):7.3f}" for n, m in pairs))
    print("\n=== held curves (chunk, arb) ===")
    for r in rows:
        name = r["file"].replace("results_v12b_", "").replace(".json", "")
        print(f"{name:26s} {[(c, round(a, 3)) for c, a, _ in r['held_curve']]}")


if __name__ == "__main__":
    main()
