"""把 WikiExtractor 格式的 jsonl 转成纯文本语料（流式，带进度）。"""
import argparse
import json
import re
import time
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=80)
    args = ap.parse_args()
    t0 = time.time()
    kept = chars = lines = 0
    with open(args.jsonl, encoding="utf-8", errors="ignore") as f, \
            open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            lines += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            text = d.get("content") or d.get("text") or ""
            if not text:
                continue
            paras = [p.strip() for p in re.split(r"\n{2,}", text) if CJK.search(p)]
            if not paras:
                continue
            clean = "\n\n".join(paras)
            if len(clean) < args.min_chars:
                continue
            w.write(clean + "\n\n")
            kept += 1
            chars += len(clean)
            if kept % 100000 == 0:
                print(f"  {kept} articles, {chars/1e6:.1f}M chars, {time.time()-t0:.0f}s",
                      flush=True)
    print(json.dumps({"jsonl": args.jsonl, "lines": lines, "kept": kept,
                      "chars": chars, "seconds": round(time.time()-t0, 1)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
