"""用 Qwen tokenizer 把纯文本语料编码成 uint32 token ids（memmap 训练用）。

用法：python tokenize_corpus.py --text E:/PycharmProject/data/zhwiki_full.txt \
        --out-prefix E:/PycharmProject/data/zhwiki_qwen --batch-lines 2000
产出：<prefix>_u32.bin, <prefix>_stats.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

TOK = "E:/PycharmProject/EvoC-CTOS-T/T-MODEL/Qwen/Qwen3-1.7B/tokenizer.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--tokenizer", default=TOK)
    ap.add_argument("--batch-lines", type=int, default=2000)
    ap.add_argument("--max-lines", type=int, default=0)
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    out = Path(args.out_prefix + "_u32.bin")
    t0 = time.time()
    n_tok = 0
    n_lines = 0
    buf_lines = []
    with open(args.text, encoding="utf-8", errors="ignore") as f, open(out, "wb") as w:
        def flush():
            nonlocal n_tok
            if not buf_lines:
                return
            encs = tok.encode_batch(buf_lines)
            ids = np.concatenate([np.asarray(e.ids, dtype=np.uint32) for e in encs])
            w.write(ids.tobytes())
            n_tok += len(ids)
            if n_tok % 20_000_000 < len(ids):
                print(f"  {n_tok/1e6:.0f}M tokens, {time.time()-t0:.0f}s", flush=True)
        for line in f:
            n_lines += 1
            buf_lines.append(line)
            if len(buf_lines) >= args.batch_lines:
                flush()
                buf_lines = []
            if args.max_lines and n_lines >= args.max_lines:
                break
        flush()
    stats = {"text": args.text, "tokens": n_tok, "lines": n_lines,
             "seconds": round(time.time() - t0, 1),
             "tokens_per_sec": round(n_tok / max(1, time.time() - t0), 1),
             "bin": str(out), "bytes": out.stat().st_size}
    Path(args.out_prefix + "_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
