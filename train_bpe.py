"""在我们自己的语料上训练 BPE tokenizer（16k，不依赖 Qwen）。"""
import argparse
import json
import time
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="E:/PycharmProject/data/zhwiki_full.txt")
    ap.add_argument("--out", default="E:/PycharmProject/data/bpe16k/tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--sample-mb", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    limit = args.sample_mb * 1024 * 1024
    n = 0
    lines = []
    with open(args.text, encoding="utf-8", errors="ignore") as f:
        for line in f:
            lines.append(line)
            n += len(line.encode("utf-8"))
            if n >= limit:
                break
    print(f"sample: {len(lines)} lines, {n/1e6:.1f} MB, {time.time()-t0:.1f}s", flush=True)
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size,
                                  special_tokens=["<unk>", "<pad>"],
                                  show_progress=True)
    tok.train_from_iterator(lines, trainer=trainer)
    tok.save(str(out))
    stats = {"text": args.text, "sample_lines": len(lines), "sample_mb": round(n/1e6, 1),
             "vocab_size": tok.get_vocab_size(), "out": str(out),
             "seconds": round(time.time()-t0, 1)}
    (out.parent / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
