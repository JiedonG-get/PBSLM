"""从 ModelScope 拉取中文维基子集，转成训练用纯文本语料。

不下载整个 2.4GB：流式读取前 MAX_MB 字节，逐行解析 WikiExtractor 格式
({"id","url","title","content"})，保留中文内容，输出：
  E:/PycharmProject/data/zhwiki_subset.txt     （纯文本语料，纯字符流）
  E:/PycharmProject/data/zhwiki_subset.json    （统计）
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

URL = ("https://www.modelscope.cn/api/v1/datasets/caoaolong/zhwiki/repo"
       "?Revision=master&FilePath=zhwiki_dataset.jsonl")
OUT_DIR = Path("E:/PycharmProject/data")
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--max-mb", type=int, default=120)
_ap.add_argument("--out", default="zhwiki_subset.txt")
_args, _ = _ap.parse_known_args()
MAX_MB = _args.max_mb
CJK = re.compile(r"[\u4e00-\u9fff]")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_txt = OUT_DIR / _args.out
    out_json = OUT_DIR / (_args.out.replace(".txt", ".json"))
    heads = {"User-Agent": "Mozilla/5.0"}
    t0 = time.time()
    req = urllib.request.Request(URL, headers=heads)
    n_bytes = 0
    lines = 0
    kept = 0
    chars = 0
    buf = b""
    with urllib.request.urlopen(req, timeout=60) as r, open(out_txt, "w", encoding="utf-8") as f:
        limit = MAX_MB * 1024 * 1024
        while n_bytes < limit:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            n_bytes += len(chunk)
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                lines += 1
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                text = d.get("content") or d.get("text") or ""
                if not text:
                    continue
                # 清理：去掉过多空行，保留含中文的段落
                paras = [p.strip() for p in re.split(r"\n{2,}", text) if CJK.search(p)]
                if not paras:
                    continue
                clean = "\n\n".join(paras)
                if len(clean) < 80:
                    continue
                f.write(clean + "\n\n")
                kept += 1
                chars += len(clean)
            if n_bytes % (20 * 1024 * 1024) < (1 << 20):
                print(f"  {n_bytes/1e6:.0f} MB, kept={kept}, chars={chars}", flush=True)
    stats = {"url": URL, "downloaded_mb": round(n_bytes / 1e6, 1), "lines": lines,
             "kept_articles": kept, "chars": chars, "seconds": round(time.time() - t0, 1),
             "out": str(out_txt)}
    out_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
