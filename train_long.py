"""长训练编排：分阶段（warm restart）继续预训练，直到 held 达标或平台期。

决策：数据集不切——完整中文维基 738.6M 字符，模型才看过 0.17%，瓶颈是训练量不是数据量；
先在同一语料上把 LM 训到收敛，之后需要生成/对话泛化时再混入其它语料（replay 保留旧分布）。

用法：python train_long.py --init ckpt_v15_lm_full.pt --phases 8000x1.5e-4,8000x1e-4,...
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
DATA = "E:/PycharmProject/data/zhwiki_full_u16.bin"
VOCAB = "E:/PycharmProject/data/zhwiki_full_vocab.json"


def ckpt_path(depth, tag):
    return OUT / f"ckpt_v12b_d{depth}_n96m32_tv_ratio{tag}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(OUT / "ckpt_v12b_d6_n96m32_tv_ratio_long_p0.pt"))
    ap.add_argument("--phases", default="8000x1e-4,8000x7e-5,8000x5e-5,8000x4e-5,8000x3e-5")
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--pretrain-clip", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--target-held", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--patience-improve", type=float, default=0.02)
    args = ap.parse_args()
    phases = []
    for p in args.phases.split(","):
        steps, lr = p.split("x")
        phases.append((int(steps), float(lr)))
    prev = args.init
    history = []
    best = 1e9
    stale = 0
    for i, (steps, lr) in enumerate(phases):
        tag = f"_long_p{i}"
        out_ckpt = ckpt_path(args.depth, tag)
        cmd = [
            sys.executable, str(OUT / "run_online.py"),
            "--data-ids", DATA, "--vocab-json", VOCAB,
            "--k", "30", "--h", "0", "--depth", str(args.depth), "--act-x", "0.2",
            "--e-n", "10", "--e-depth", "1", "--e-lr-mult", "1.0",
            "--init-ckpt", str(prev), "--steps", str(steps), "--chunks", "0",
            "--batch", str(args.batch), "--eval-windows", str(args.eval_windows),
            "--sweep", "96x32", "--commit-beta", "0.3",
            "--pretrain-clip", str(args.pretrain_clip), "--tag", tag,
        ]
        log_path = OUT / f"log_long_p{i}.txt"
        t0 = time.time()
        print(f"[long] phase {i}: steps={steps} lr={lr} init={Path(prev).name}", flush=True)
        with open(log_path, "w", encoding="utf-8") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        res_json = OUT / f"results_v12_d{args.depth}_n96m32_tv_ratio_a_side{tag}.json"
        held = None
        if res_json.exists():
            r = json.load(open(res_json, encoding="utf-8"))
            held = r["held_after"]["arb"]
        rec = {"phase": i, "steps": steps, "lr": lr, "held": held,
               "seconds": round(time.time() - t0, 1), "ckpt": str(out_ckpt)}
        history.append(rec)
        print("[long]", json.dumps(rec, ensure_ascii=False), flush=True)
        json.dump(history, open(OUT / "long_training.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        if held is None:
            break
        if held <= args.target_held:
            print(f"[long] target reached: held={held:.4f} <= {args.target_held}", flush=True)
            break
        if held < best - args.patience_improve:
            best = held
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print("[long] plateau, stop", flush=True)
                break
        prev = out_ckpt if out_ckpt.exists() else prev
    print("[long] done", flush=True)


if __name__ == "__main__":
    main()
