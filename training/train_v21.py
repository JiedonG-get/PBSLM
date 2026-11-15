"""v2.1 训练脚本（Phase 0/1 基准用）。

特点：
- 配置驱动（JSON），Baseline A1/A2 只差 codebook 字段；
- 单路径连续向量 + InfoNCE（Phase 1 实际为 sampled-negative：1 个正样本 + neg_n 个采样负例；不做全词表 softmax，也不含 batch 内负例）；
- checkpoint 保存 model/optimizer/step/config/随机数状态 + 固定码本 hash；
- resume 跳过 field 初始化（v2.1 单路径无 field，逻辑保留给后续 dual 模型）；
- 评估：top1/top5、频率分组、平均余弦、上下文交换敏感性、困难负样本胜率。

Phase 0 阶段只做代码整理；Phase 0.5 的 smoke 命令见 experiments/phase0/PLAN_v2_1.md。
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from models.model_v21 import SinglePathContinuousLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ids(path, dtype="uint32", hold=0.15):
    arr = np.memmap(path, dtype=np.uint32 if dtype == "uint32" else np.uint16, mode="r")
    cut = int(len(arr) * (1 - hold))
    train = np.asarray(arr[:cut])
    held = np.asarray(arr[cut:])
    return train, held


def build_model(cfg):
    m = cfg["model"]
    arch = m.get("arch", "single")
    if arch != "single":
        raise NotImplementedError("Phase 0 只实现 baseline 单路径；dual 在 Phase 2 加入")
    return SinglePathContinuousLM(
        vocab=m["vocab"], d=m["d"], heads=m["heads"], k=m["k"], depth=m["depth"],
        codebook=m["codebook"], codebook_seed=m.get("codebook_seed", 1234),
        codebook_init=m.get("codebook_init", "normal"),
    )


def trainable_parameters(model):
    """只返回真正 requires_grad 的参数（v2.1 会冻结 BlockBank.comm 这类前向未用的参数）。"""
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(model, cfg):
    o = cfg["optim"]
    params = trainable_parameters(model)
    if o["type"] == "adamw":
        return torch.optim.AdamW(params, lr=o["lr"], betas=tuple(o.get("betas", [0.9, 0.999])),
                                 eps=o.get("eps", 1e-8), weight_decay=o.get("weight_decay", 0.0))
    if o["type"] == "adam":
        return torch.optim.Adam(params, lr=o["lr"], betas=tuple(o.get("betas", [0.9, 0.999])))
    if o["type"] == "sgd":
        return torch.optim.SGD(params, lr=o["lr"], momentum=o.get("momentum", 0.9))
    raise ValueError(f"unknown optimizer {o['type']}")


def codebook_hash(model):
    return hashlib.sha1(model.code_table.weight.detach().cpu().numpy().tobytes()).hexdigest()


def ar_batch(data, n_in, ar_m, batch, device, gen):
    starts = torch.randint(0, len(data) - n_in - ar_m - 1, (batch,), generator=gen)
    xs, ys = [], []
    for j in range(ar_m):
        xs.append(torch.stack([torch.from_numpy(data[s + j:s + j + n_in].copy()) for s in starts]))
        ys.append(torch.stack([torch.from_numpy(np.asarray([data[s + n_in + j]])) for s in starts]).squeeze(1))
    x = torch.stack(xs, 1).reshape(-1, n_in).to(device).long()
    y = torch.stack(ys, 1).reshape(-1).to(device).long()
    return x, y


def info_nce_loss(model, z, y, cfg, gen):
    lc = cfg["loss"]
    tau = lc["tau"]
    tgt = model.target_embed(y)
    P = z.shape[0]
    pos = (z * tgt).sum(-1, keepdim=True) / tau
    neg_n = lc.get("neg_n", 128)
    nid = torch.randint(0, model.vocab, (P, neg_n), device=z.device, generator=gen)
    neg = model.target_embed(nid)
    negl = torch.einsum("pd,pnd->pn", z, neg) / tau
    if lc.get("exclude_positive", True):
        negl = negl.masked_fill(nid == y[:, None], -torch.finfo(negl.dtype).max)
    logits = torch.cat([pos, negl], dim=1)
    labels = torch.zeros(P, dtype=torch.long, device=z.device)
    loss = F.cross_entropy(logits, labels)
    ppos = torch.exp(pos - torch.logsumexp(logits, 1, keepdim=True)).mean()
    return loss, float(ppos.detach())


@torch.no_grad()
def evaluate(model, data, cfg, freq, device):
    """评估指标口径：
    - other_ctx_acc：随机另一条上下文 + **它自己的正确目标** 的 top-1；
      它衡量“换上下文后的绝对准确率”，不是配对上下文交换的准确率变化。
    - 上下文敏感性主要看：
      * ctx_change_rate：原上下文 vs 随机上下文预测变化率；
      * ctrl_keep_last_acc / ctrl_keep_last_change：保留最后一个 token、替换其余后的准确率/变化率。
    - hard_neg_win：固定困难负样本集合上的胜率（见下方来源注释）。
    """
    ev = cfg["eval"]
    n_in = cfg["data"]["n_in"]
    ar_m = cfg["data"]["ar_m"]
    windows = ev["windows"]
    model.eval()
    starts = np.linspace(0, len(data) - n_in - ar_m - 2, windows).astype(int).tolist()
    top1 = top5 = 0.0
    cos_sum = 0.0
    ctx_change = 0.0
    other_ctx_acc = 0.0
    ctrl_keep_last_acc = 0.0
    ctrl_keep_last_change = 0.0
    hard_wins = []
    bucket = {"high": [], "mid": [], "low": [], "rare": []}
    rng = np.random.default_rng(cfg["seed"] + 7)
    E = F.normalize(model.code_table.weight.float(), dim=-1)

    # 固定困难负样本池：训练集最高频的 top-N token（与模型自己的全词表排序无关）。
    # 每个目标 token 用固定 seed(hard_neg_seed + token_id) 从池中抽 K 个，排除目标 token。
    hard_pool = np.argsort(-freq)[:int(ev.get("hard_neg_pool", 512))]
    hard_k = int(ev.get("hard_neg_k", 16))
    hard_seed = int(ev.get("hard_neg_seed", cfg["seed"] + 23))

    def hard_negatives(t):
        pool = [int(x) for x in hard_pool if int(x) != int(t)]
        if not pool:
            return np.empty(0, dtype=np.int64)
        gg = np.random.default_rng(hard_seed + int(t))
        return gg.choice(pool, size=min(hard_k, len(pool)), replace=False).astype(np.int64)

    for s in starts:
        xs, ys = [], []
        for j in range(ar_m):
            xs.append(torch.from_numpy(data[s + j:s + j + n_in].copy()))
            ys.append(int(data[s + n_in + j]))
        x = torch.stack(xs, 0).unsqueeze(0).reshape(-1, n_in).to(device).long()
        y = torch.tensor(ys, device=device)
        z = model(x)                                  # (ar_m,d)
        sim = z.float() @ E.t()                       # (ar_m,V)
        pred1 = sim.argmax(-1)
        top1 += float((pred1 == y).float().mean())
        topk = sim.topk(5, dim=-1).indices
        top5 += float((topk == y[:, None]).any(-1).float().mean())
        tgt = model.target_embed(y).float()
        cos_sum += float(F.cosine_similarity(z.float(), tgt, dim=-1).mean())

        # 独立困难负样本：固定池 + 固定 seed；只在这 K 个候选上算胜率
        for idx, tok in enumerate(y.tolist()):
            nids = hard_negatives(tok)
            if len(nids) == 0:
                continue
            nid_t = torch.as_tensor(nids, device=device, dtype=torch.long)
            s_t = sim[idx, tok]
            s_n = sim[idx, nid_t]
            hard_wins.append(float((s_t > s_n).float().mean()))

        # 上下文交换（目标匹配版）：另一个随机上下文配它自己的正确目标
        s2 = int(rng.integers(0, len(data) - n_in - ar_m - 2))
        xs2, ys2 = [], []
        for j in range(ar_m):
            xs2.append(torch.from_numpy(data[s2 + j:s2 + j + n_in].copy()))
            ys2.append(int(data[s2 + n_in + j]))
        x2 = torch.stack(xs2, 0).unsqueeze(0).reshape(-1, n_in).to(device).long()
        y2 = torch.tensor(ys2, device=device)
        pred2 = (model(x2).float() @ E.t()).argmax(-1)
        ctx_change += float((pred2 != pred1).float().mean())
        other_ctx_acc += float((pred2 == y2).float().mean())

        # 受控上下文替换：保留最后一个 token，其余位置替换成随机 token
        xc = x.clone()
        xc[:, :-1] = torch.randint(0, model.vocab, (ar_m, n_in - 1),
                                   device=device, dtype=torch.long)
        predc = (model(xc).float() @ E.t()).argmax(-1)
        ctrl_keep_last_acc += float((predc == y).float().mean())
        ctrl_keep_last_change += float((predc != pred1).float().mean())

        # 频率分组
        for p, t in zip(pred1.tolist(), y.tolist()):
            f = freq[t]
            if f >= 0.01:
                bucket["high"].append(float(p == t))
            elif f >= 0.001:
                bucket["mid"].append(float(p == t))
            elif f >= 1e-4:
                bucket["low"].append(float(p == t))
            else:
                bucket["rare"].append(float(p == t))
    model.train()
    return {
        "held_top1": top1 / windows, "held_top5": top5 / windows,
        "cos": cos_sum / windows,
        "hard_neg_win": float(np.mean(hard_wins)) if hard_wins else None,
        "hard_neg_source": "fixed pool=top-512 frequent train tokens; K=16 per target; "
                           "seed=hard_neg_seed+token_id; target excluded",
        "ctx_change_rate": ctx_change / windows,
        "other_ctx_acc": other_ctx_acc / windows,
        "ctrl_keep_last_acc": ctrl_keep_last_acc / windows,
        "ctrl_keep_last_change": ctrl_keep_last_change / windows,
        "freq_bucket": {k: (float(np.mean(v)) if v else None) for k, v in bucket.items()},
        "freq_bucket_n": {k: len(v) for k, v in bucket.items()},
    }

def save_checkpoint(path, model, opt, step, cfg, neg_gen, data_gen):
    ck = {
        "format": "v2.1",
        "step": int(step),
        "config": cfg,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_np": np.random.get_state(),
        "neg_gen": neg_gen.get_state(),
        "data_gen": data_gen.get_state(),
        "codebook_hash": codebook_hash(model),   # 每次保存时重新计算
        "codebook_mode": model.codebook,
    }
    torch.save(ck, path)


ALLOWED_RESUME_DIFF = {
    ("training", "steps"),
    ("logging", "metrics_jsonl"),
    ("logging", "log_every"),
    ("logging", "eval_every"),
    ("checkpoint", "save"),
    ("checkpoint", "save_every"),
    ("device",),
}


def flatten_cfg(d, prefix=()):
    out = {}
    for k, v in d.items():
        key = prefix + (k,)
        if isinstance(v, dict):
            out.update(flatten_cfg(v, key))
        else:
            out[key] = v
    return out


def validate_resume_config(saved_cfg, current_cfg):
    saved = flatten_cfg(saved_cfg)
    current = flatten_cfg(current_cfg)
    diffs = []
    for key in sorted(set(saved) | set(current)):
        if key in ALLOWED_RESUME_DIFF:
            continue
        if saved.get(key) != current.get(key):
            diffs.append((key, saved.get(key), current.get(key)))
    if diffs:
        detail = "; ".join(f"{'.'.join(k)}: ckpt={a!r} config={b!r}" for k, a, b in diffs[:8])
        raise ValueError(f"resume 配置不兼容，拒绝恢复：{detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--steps", type=int, default=-1, help="覆盖 config 的 steps（Phase 0.5 smoke 用）")
    ap.add_argument("--save", default="", help="覆盖 config 的 save 路径")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    device = torch.device(args.device or cfg.get("device", DEV))
    set_seed(cfg["seed"])
    train_data, held_data = load_ids(cfg["data"]["ids"], cfg["data"].get("dtype", "uint32"),
                                     cfg["data"].get("hold", 0.15))
    print(f"train={len(train_data)} held={len(held_data)} device={device}", flush=True)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in trainable_parameters(model))
    print(f"arch={cfg['model'].get('arch','single')} params(trainable)={n_params/1e6:.2f}M "
          f"codebook={model.codebook}", flush=True)

    opt = build_optimizer(model, cfg)
    neg_gen = torch.Generator(device=device)
    neg_gen.manual_seed(cfg["seed"] + 13)
    start_step = 0
    gen = torch.Generator()  # CPU generator：只用于数据窗口起点
    gen.manual_seed(cfg["seed"] + 17)  # 与 neg_gen 分开，避免互相扰动

    if args.resume:
        ck = torch.load(args.resume, weights_only=False, map_location="cpu")
        if ck.get("format") != "v2.1":
            raise ValueError("checkpoint format 不是 v2.1")
        if "config" not in ck:
            raise ValueError("checkpoint 缺少 config，无法校验 resume 兼容性")
        validate_resume_config(ck["config"], cfg)
        if ck.get("codebook_mode") != model.codebook:
            raise ValueError("resume 码本模式与 config 不一致")
        if model.codebook == "fixed" and ck.get("codebook_hash") != codebook_hash(model):
            raise ValueError("fixed codebook hash 不一致，拒绝 resume（目标空间已变）")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step = int(ck["step"])
        torch.set_rng_state(ck["rng_torch"])
        if ck.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        np.random.set_state(ck["rng_np"])
        neg_gen.set_state(ck["neg_gen"])
        if "data_gen" in ck:
            gen.set_state(ck["data_gen"])
        print(f"resumed step={start_step} from {args.resume}", flush=True)

    # v2.1 单路径没有 field；这段保留给 dual 模型，resume 时必须跳过
    if start_step == 0 and hasattr(model, "field"):
        with torch.no_grad():
            last = model.field.net[-1]
            last.bias[model.d:] = -2.0
            last.weight.mul_(0.1)
        print("initialized field (fresh model only)", flush=True)

    mcfg = cfg["model"]
    steps = args.steps if args.steps > 0 else cfg["training"]["steps"]
    batch = cfg["training"]["batch"]
    n_in = cfg["data"]["n_in"]
    ar_m = cfg["data"]["ar_m"]
    save_path = args.save or cfg["checkpoint"]["save"]
    save_every = cfg["checkpoint"].get("save_every", 0)
    eval_every = cfg["logging"].get("eval_every", 1000)
    log_every = cfg["logging"].get("log_every", 100)
    metrics_path = cfg["logging"].get("metrics_jsonl", "")
    freq = np.bincount(train_data, minlength=mcfg["vocab"]).astype(np.float64)
    freq = freq / freq.sum()
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(start_step, steps):
        step_t0 = time.time()
        opt.zero_grad(set_to_none=True)
        x, y = ar_batch(train_data, n_in, ar_m, batch, device, gen)
        z = model(x)
        loss, ppos = info_nce_loss(model, z, y, cfg, neg_gen)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters(model), cfg["optim"].get("grad_clip", 1.0))
        opt.step()
        lr = opt.param_groups[0]["lr"]
        rec = {"step": step, "loss": round(float(loss.detach()), 4),
               "ppos": round(ppos, 4), "grad_norm": round(float(gnorm), 4),
               "lr": lr, "sec": round(time.time() - t0, 1),
               "dt": round(time.time() - step_t0, 4)}
        if step % log_every == 0:
            print("STEP " + json.dumps(rec, ensure_ascii=False), flush=True)
        if eval_every > 0 and (step + 1) % eval_every == 0:
            met = evaluate(model, held_data, cfg, freq, device)
            rec["eval"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()}
            print("EVAL " + json.dumps(rec["eval"], ensure_ascii=False), flush=True)
        if metrics_path:
            with open(metrics_path, "a", encoding="utf-8") as mf:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if save_every > 0 and save_path and (step + 1) % save_every == 0:
            save_checkpoint(save_path, model, opt, step + 1, cfg, neg_gen, gen)
            print(f"saved {save_path} @step{step+1}", flush=True)
    if save_path:
        save_checkpoint(save_path, model, opt, steps, cfg, neg_gen, gen)
        print(f"saved final {save_path}", flush=True)
    if torch.cuda.is_available():
        print(f"PEAK_MEM allocated_gib={torch.cuda.max_memory_allocated()/1024**3:.3f} "
              f"reserved_gib={torch.cuda.max_memory_reserved()/1024**3:.3f}", flush=True)


if __name__ == "__main__":
    main()
