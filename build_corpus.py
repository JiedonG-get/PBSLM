"""大语料 -> 词表 + uint16 ids 二进制（memmap 训练用）。

用法：python build_corpus.py --text E:/PycharmProject/data/zhwiki_full.txt \
        --out-prefix E:/PycharmProject/data/zhwiki_full
产出：<prefix>_vocab.json, <prefix>_u16.bin, <prefix>_stats.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    text_path = Path(args.text)
    t0 = time.time()
    # pass 1: 词表
    vocab = set()
    lines = 0
    with open(text_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            lines += 1
            vocab.update(line)
            if lines % 2_000_000 == 0:
                print(f"  vocab pass {lines/1e6:.0f}M lines, {len(vocab)} chars, "
                      f"{time.time()-t0:.0f}s", flush=True)
    vocab = sorted(vocab)
    if len(vocab) > 65535:
        raise SystemExit(f"vocab too large for uint16: {len(vocab)}")
    stoi = {c: i for i, c in enumerate(vocab)}
    vocab_path = Path(args.out_prefix + "_vocab.json")
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"vocab={len(vocab)} saved {vocab_path} ({time.time()-t0:.0f}s)", flush=True)

    # pass 2: 编码成 uint16 二进制
    bin_path = Path(args.out_prefix + "_u16.bin")
    n = 0
    with open(text_path, encoding="utf-8", errors="ignore") as f, open(bin_path, "wb") as w:
        for line in f:
            arr = np.fromiter((stoi[c] for c in line), dtype=np.uint16, count=len(line))
            w.write(arr.tobytes())
            n += len(arr)
            if n % 50_000_000 < len(line) and n >= 50_000_000:
                print(f"  encoded {n/1e6:.0f}M chars, {time.time()-t0:.0f}s", flush=True)
    stats = {"text": str(text_path), "vocab": len(vocab), "chars": n,
             "bin": str(bin_path), "bytes": bin_path.stat().st_size,
             "seconds": round(time.time()-t0, 1)}
    Path(args.out_prefix + "_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
