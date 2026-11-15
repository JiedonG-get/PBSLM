"""并行 Range 下载器（ModelScope CDN 支持 206）。用线程分块下载后按序拼接。

用法：python dl_parallel.py --out E:/PycharmProject/data/zhwiki_dataset.jsonl --total 2462455277 --threads 16
"""
import argparse
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = ("https://www.modelscope.cn/api/v1/datasets/caoaolong/zhwiki/repo"
       "?Revision=master&FilePath=zhwiki_dataset.jsonl")
HEADS = {"User-Agent": "Mozilla/5.0"}


def fetch_range(start, end, path):
    req = urllib.request.Request(URL, headers={**HEADS, "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=32)
    ap.add_argument("--test-mb", type=int, default=0, help=">0 只测速：下这么多MB就退出")
    args = ap.parse_args()
    total = min(args.total, args.test_mb * 1024 * 1024) if args.test_mb else args.total
    size = args.chunk_mb * 1024 * 1024
    ranges = [(s, min(s + size - 1, total - 1)) for s in range(0, total, size)]
    parts = [args.out + f".part{i:03d}" for i in range(len(ranges))]
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(fetch_range, s, e, p): i for i, ((s, e), p) in enumerate(zip(ranges, parts))}
        for fu in as_completed(futs):
            done += 1
            if done % max(1, len(ranges) // 10) == 0:
                mb = sum(os.path.getsize(p) for p in parts if os.path.exists(p)) / 1e6
                el = time.time() - t0
                print(f"  {mb:.0f}/{total/1e6:.0f} MB  {el:.0f}s  {mb/el:.2f} MB/s", flush=True)
    got = sum(os.path.getsize(p) for p in parts)
    print(f"downloaded {got/1e6:.1f} MB in {time.time()-t0:.1f}s")
    if args.test_mb:
        for p in parts:
            os.remove(p)
        return
    assert got == total, (got, total)
    with open(args.out, "wb") as f:
        for p in parts:
            with open(p, "rb") as g:
                while True:
                    b = g.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
    for p in parts:
        os.remove(p)
    print("saved", args.out, os.path.getsize(args.out))


if __name__ == "__main__":
    main()
