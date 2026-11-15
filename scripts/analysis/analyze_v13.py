"""汇总 v13（30 并行块 × depth6，活动度 20%，信息流动一致性，震撼值）测试。"""
import glob
import json
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "results_v12_d6_*_v13*.json"))):
        if "_v13smoke" in p:
            continue
        r = json.load(open(p, encoding="utf-8"))
        cfg = r["config"]
        rec = r["online"]["records"]

        def arr(k):
            return np.array(rec.get(k, []), dtype=float)

        rows.append({
            "file": os.path.basename(p),
            "tag": "shock" if cfg.get("shock_beta", 0) > 0 else "noshock",
            "k": cfg["k"], "depth": cfg["depth"], "act_x": cfg.get("act_x"),
            "shock_beta": cfg.get("shock_beta"), "batch": cfg.get("batch"),
            "steps": cfg.get("steps"), "chunks": cfg.get("chunks"),
            "params": r["model_params"],
            "pretrain_seconds": r["pretrain"]["seconds"],
            "loss_first": r["pretrain"]["loss_curve"][0],
            "loss_last": r["pretrain"]["loss_curve"][-1],
            "shock_mean": float(np.mean(r["pretrain"].get("shock_curve", [0]))),
            "active_mean": float(np.mean(r["pretrain"].get("active_curve", [0]))),
            "active_max": int(np.max(r["pretrain"].get("active_curve", [0]))),
            "held_before": r["held_before"]["main"],
            "held_after": r["held_after"],
            "d_arb": round(r["held_after"]["arb"] - r["held_before"]["main"]["arb"], 4),
            "stream_first500": float(np.mean(arr("ce_arb")[:500])),
            "stream_last500": float(np.mean(arr("ce_arb")[-500:])),
            "act_mean": float(arr("act").mean()),
            "act_max": float(arr("act").max()),
            "online_active_mean": float(arr("active").mean()),
            "online_active_max": int(arr("active").max()),
            "online_shock_mean": float(arr("shock").mean()),
            "g_mean": float(arr("g").mean()),
            "g_std": float(arr("g").std()),
            "held_curve": [(h["chunk"], h["held"]["arb"]) for h in r["online"]["held_curve"]],
            "sweep_after": r["sweep_after"],
            "ckpt": r.get("checkpoint"),
        })
    return rows


def main():
    rows = load()
    if not rows:
        print("no v13 results")
        return
    json.dump(rows, open(os.path.join(OUT, "v13_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    for r in rows:
        print(f"== {r['tag']}  k={r['k']} depth={r['depth']} act_x={r['act_x']} "
              f"shock_beta={r['shock_beta']} batch={r['batch']}")
        print(f"   params={r['params']['total']/1e6:.1f}M online={r['params']['online']/1e6:.2f}M "
              f"bank={r['params']['bank']/1e6:.1f}M flow={r['params']['flow']/1e3:.1f}k")
        print(f"   pretrain {r['pretrain_seconds']}s loss {r['loss_first']}->{r['loss_last']} "
              f"shock_mean={r['shock_mean']:.2f} active mean/max={r['active_mean']:.1f}/{r['active_max']}")
        print(f"   held arb {r['held_before']['arb']:.3f} -> {r['held_after']['arb']:.3f} "
              f"({r['d_arb']:+.3f})   stream {r['stream_first500']:.3f} -> {r['stream_last500']:.3f}")
        print(f"   online: act mean={r['act_mean']:.3f} max={r['act_max']:.3f} "
              f"active mean/max={r['online_active_mean']:.1f}/{r['online_active_max']} "
              f"shock={r['online_shock_mean']:.2f} g={r['g_mean']:.3f}±{r['g_std']:.3f}")
        print("   held curve:", [(c, round(a, 3)) for c, a in r["held_curve"]])
        print("   sweep after:", [(s["n_in"], s["n_out"], s["arb"]) for s in r["sweep_after"]])


if __name__ == "__main__":
    main()
