"""检查 Phase 1.1 codebook 对照：初始化数值一致、参数量与 tokenizer unk id。

只构造模型，不训练、不前向。
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_v21 import build_model, trainable_parameters  # noqa: E402


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        if isinstance(v, torch.Tensor):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    cfg_a = json.load(open("configs/ctrl_codebook_trainable.json", encoding="utf-8"))
    cfg_b = json.load(open("configs/ctrl_codebook_frozen.json", encoding="utf-8"))
    # 两个配置在同一 seed 下独立构造，模拟两次独立训练进程
    torch.manual_seed(cfg_a["seed"])
    a = build_model(cfg_a)
    torch.manual_seed(cfg_b["seed"])
    b = build_model(cfg_b)
    sa, sb = a.state_dict(), b.state_dict()
    keys_equal = sorted(sa) == sorted(sb)
    values_equal = keys_equal and all(torch.equal(sa[k], sb[k]) for k in sa)
    total_a = sum(p.numel() for p in a.parameters())
    total_b = sum(p.numel() for p in b.parameters())
    train_a = sum(p.numel() for p in trainable_parameters(a))
    train_b = sum(p.numel() for p in trainable_parameters(b))
    try:
        from tokenizers import Tokenizer
        t = Tokenizer.from_file("tokenizer/bpe16k_tokenizer.json")
        tok = {"vocab": t.get_vocab_size(), "id0": t.id_to_token(0),
               "unk_id": t.token_to_id("<unk>")}
    except Exception as e:
        tok = {"error": str(e)}
    out = {
        "configs": {"trainable": "configs/ctrl_codebook_trainable.json",
                    "frozen": "configs/ctrl_codebook_frozen.json"},
        "state_keys_equal": keys_equal,
        "all_initial_values_equal": values_equal,
        "state_hash_equal": state_hash(a) == state_hash(b),
        "state_hash": state_hash(a),
        "total_params": {"trainable": total_a, "frozen": total_b},
        "trainable_params": {"trainable": train_a, "frozen": train_b},
        "tokenizer": tok,
    }
    path = Path("experiments/phase1_1/metrics/control_init_check.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
