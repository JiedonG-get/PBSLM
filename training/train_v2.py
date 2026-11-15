"""v2 本地 smoke：自训 BPE + 连续向量输出 + 扰动场；fp16 + Adafactor（省显存）。

用法示例（先跑通，不求质量）：
  python train_v2.py --ids E:/PycharmProject/data/bpe_smoke_u32.bin --ids-dtype uint32 \
      --steps 100 --batch 1 --n-in 16 --n-out 8
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from models.model_v2 import PBSLMv2

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def r4(o):
    """日志里所有浮点数统一保留 4 位小数（递归处理 dict/list/tuple）。"""
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, dict):
        return {k: r4(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [r4(v) for v in o]
    return o


def load_ids(path, dtype, hold=0.15):
    arr = np.memmap(path, dtype=dtype, mode="r")
    n = len(arr)
    cut = int(n * (1 - hold))
    train = torch.from_numpy(np.asarray(arr[:cut]))
    held = torch.from_numpy(np.asarray(arr[cut:]))
    return train, held



@torch.no_grad()
def eval_decode_acc(model, data, n_in, n_out, windows=16, act=0.2):
    model.eval()
    vals = []
    starts = np.linspace(0, len(data) - n_in - n_out - 1, windows).astype(int).tolist()
    for s in starts:
        x = data[s:s + n_in].unsqueeze(0).to(DEV).long()
        y = data[s + n_in:s + n_in + n_out].unsqueeze(0).to(DEV).long()
        out = model(x, n_out, act=act, sample_noise=False)
        sim = model.decode_nearest(out["z_pert"][0])
        pred = sim.argmax(-1)
        vals.append(float((pred == y[0]).float().mean()))
    model.train()
    return float(np.mean(vals))


@torch.no_grad()
def eval_ar_acc(model, data, n_in, ar_m=8, windows=64, act=0.2):
    model.eval()
    vals = []
    starts = np.linspace(0, len(data) - n_in - ar_m - 1, windows).astype(int).tolist()
    for s in starts:
        xs, ys = [], []
        for j in range(ar_m):
            xs.append(data[s + j:s + j + n_in])
            ys.append(data[s + n_in + j])
        x = torch.stack(xs, 0).unsqueeze(0).reshape(-1, n_in).to(DEV).long()
        y = torch.stack(ys, 0).unsqueeze(0).reshape(-1, 1).to(DEV).long()
        out = model(x, 1, act=act, sample_noise=False)
        sim = model.decode_nearest(out["z_pert"][:, 0])
        pred = sim.argmax(-1)
        vals.append(float((pred == y[:, 0]).float().mean()))
    model.train()
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ids-dtype", choices=["uint16", "uint32"], default="uint32")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--n-in", type=int, default=16)
    ap.add_argument("--n-out", type=int, default=8)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--act", type=float, default=0.2)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--metrics-jsonl", default="")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-windows", type=int, default=16)
    ap.add_argument("--save", default="")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--opt", choices=["adafactor", "sgd", "adam8bit", "adamw"], default="adam8bit")
    ap.add_argument("--loss", choices=["cos", "infonce", "ce"], default="infonce")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--neg", type=int, default=128)
    ap.add_argument("--field-scale", type=float, default=0.1)
    ap.add_argument("--no-mem", action="store_true")
    ap.add_argument("--bal-alpha", type=float, default=0.5)
    ap.add_argument("--frozen-code", action="store_true")
    ap.add_argument("--simple-dec", action="store_true")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--direct-only", action="store_true")
    ap.add_argument("--ar", action="store_true", help="teacher-forcing 多位置训练：每个位置用真实前文")
    ap.add_argument("--ar-m", type=int, default=8)
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--var-w", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype = np.uint32 if args.ids_dtype == "uint32" else np.uint16
    train_ids, held_ids = load_ids(args.ids, dtype)
    print(f"train={len(train_ids)} held={len(held_ids)} ids_dtype={args.ids_dtype}", flush=True)
    _cnt = np.bincount(np.asarray(train_ids, dtype=np.int64), minlength=16384).astype(np.float64)
    _freq = torch.tensor(_cnt / _cnt.sum(), device=DEV, dtype=torch.float32)
    print(f"freq top: token204={float(_freq[204]):.4f}", flush=True)

    model = PBSLMv2(vocab=16384, k=args.k, depth=args.depth, d=args.d,
                    heads=args.heads, m_max=32, act_x=args.act,
                    field_scale=args.field_scale, use_mem=not args.no_mem,
                    trainable_target=not args.frozen_code, simple_dec=args.simple_dec,
                    dense=args.dense, use_flow=not args.no_flow,
                    direct_only=args.direct_only)
    if args.bf16:
        model = model.bfloat16()
    elif not args.no_fp16:
        model = model.half()
    model = model.to(DEV)
    if args.resume:
        _ck = torch.load(args.resume, weights_only=False, map_location="cpu")
        model.load_state_dict(_ck["state"])
        print(f"resumed from {args.resume}", flush=True)
    # 扰动场初始化为小 σ（避免一开始噪声把损失顶成噪声底）
    with torch.no_grad():
        last = model.field.net[-1]
        last.bias[model.d:] = -2.0
        last.weight.mul_(0.1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_params/1e6:.1f}M fp16={not args.no_fp16}", flush=True)

    if args.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    elif args.opt == "adam8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                 eps=1e-8)
    elif args.opt == "adafactor":
        opt = torch.optim.Adafactor(model.parameters(), lr=args.lr, eps=(1e-30, 1e-3))
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    use_amp = not (args.no_fp16 or args.bf16)
    gen = torch.Generator().manual_seed(args.seed)
    span = args.n_in + args.n_out + 1
    t0 = time.time()
    _bs = args.batch * (args.ar_m if args.ar else 1)
    prev_shock = torch.zeros(_bs, 1, device=DEV)

    def loss_step(x, y, prev_shock):
        out = model(x, args.n_out, act=args.act, shock=prev_shock)
        tgt = model.target_embed(y).float()
        z = F.normalize(out["z_pert"].float(), dim=-1)
        zc = F.normalize(out["z_c"].float(), dim=-1)
        rep = F.normalize(model.a_pred(out["h_a"]).float(), dim=-1)
        L_rep = (1 - F.cosine_similarity(rep, tgt, dim=-1)).mean()

        def infonce(zs, neg_n, y_flat):
            zs = zs.reshape(-1, zs.shape[-1])
            P, D = zs.shape
            pos = (zs * tgt.reshape(P, D)).sum(-1, keepdim=True) / args.tau
            if neg_n > 0:
                nid = torch.randint(0, model.vocab, (P, neg_n), device=DEV)
                neg = model.target_embed(nid).to(zs.dtype)  # (P,neg,D) 来自可训练嵌入
                negl = torch.einsum("pd,pkd->pk", zs, neg) / args.tau
                logits = torch.cat([pos, negl], dim=1)
            else:
                logits = pos
            labels = torch.zeros(P, dtype=torch.long, device=DEV)
            ce_i = F.cross_entropy(logits, labels, reduction="none")
            # 频率平衡：高频 token 降权，防止模型塌缩到最高频 token
            w = (1.0 / (_freq[y_flat.reshape(-1)] + 1e-6)) ** args.bal_alpha
            w = w / w.mean()
            ce = (ce_i * w).mean()
            ppos = torch.exp(pos - torch.logsumexp(logits, 1, keepdim=True)).mean()
            return ce, ppos

        if args.loss == "infonce":
            L_a, ppos_a = infonce(z, args.neg, y)
            L_c, ppos_c = infonce(zc, args.neg, y)
        elif args.loss == "ce":
            # 诊断用：全词表 CE（临时，不是最终方案）
            P = z.reshape(-1, z.shape[-1]).shape[0]
            W = F.normalize(model.tok.weight.float(), dim=-1)
            la = z.reshape(P, -1).float() @ W.t() / args.tau
            lc = zc.reshape(P, -1).float() @ W.t() / args.tau
            yf = y.reshape(-1).long()
            ce_a = F.cross_entropy(la, yf, reduction="none")
            ce_c = F.cross_entropy(lc, yf, reduction="none")
            w = (1.0 / (_freq[yf] + 1e-6)) ** args.bal_alpha
            w = w / w.mean()
            L_a = (ce_a * w).mean()
            L_c = (ce_c * w).mean()
            ppos_a = ppos_c = torch.tensor(0.0, device=DEV)
        else:
            L_a = (1 - F.cosine_similarity(z, tgt, dim=-1)).mean()
            L_c = (1 - F.cosine_similarity(zc, tgt, dim=-1)).mean()
            ppos_a = ppos_c = torch.tensor(0.0, device=DEV)
        mu = out["field_mu"].float()
        log_s = out["field_log_sigma"].float()
        sigma = torch.exp(log_s)
        kl = (0.5 * (mu ** 2 + sigma ** 2 - 1 - 2 * log_s)).mean()
        # 嵌入多样性正则，防止所有 token 嵌入塌缩到同一方向
        idx = torch.randint(0, model.vocab, (1024,), device=DEV)
        E_n = F.normalize(model.tok.weight[idx], dim=-1).float()
        # 单位向量随机初始化时每维 std ≈ 1/sqrt(d)；用 sqrt(d)*std 作为多样性尺度
        L_div = F.relu(1.0 - E_n.std(0) * (model.d ** 0.5)).mean()
        D_ = z.reshape(-1, z.shape[-1]).shape[-1]
        L_var = F.relu(1.0 - z.reshape(-1, z.shape[-1]).std(0) * (D_ ** 0.5)).mean()
        L_var += F.relu(1.0 - zc.reshape(-1, zc.shape[-1]).std(0) * (D_ ** 0.5)).mean()
        loss = L_a + 0.5 * L_c + 0.1 * L_rep + 0.01 * kl + 0.1 * L_div + args.var_w * L_var
        with torch.no_grad():
            da = (1 - (z * tgt).sum(-1)).clamp(0, 2)   # 余弦距离 (B,m)
            dc = (1 - (zc * tgt).sum(-1)).clamp(0, 2)
            tau = 0.5
            known = torch.maximum(torch.exp(-da / tau), torch.exp(-dc / tau))
            S = (-torch.log(known.clamp_min(1e-3))).mean(1, keepdim=True)
        with torch.no_grad():
            mets = {
                "L_a": float(L_a.detach()), "L_c": float(L_c.detach()),
                "L_var": float(L_var.detach()),
                "ppos_a": float(ppos_a.detach()), "ppos_c": float(ppos_c.detach()),
                "L_rep": float(L_rep.detach()), "kl": float(kl.detach()),
                "L_div": float(L_div.detach()),
                "cos_a": float(F.cosine_similarity(z, tgt, dim=-1).mean()),
                "cos_c": float(F.cosine_similarity(zc, tgt, dim=-1).mean()),
                "z_norm": float(out["z_a"].float().norm(dim=-1).mean()),
                "zp_norm": float(out["z_pert"].float().norm(dim=-1).mean()),
                "mu_norm": float(mu.norm(dim=-1).mean()),
                "sigma_mean": float(sigma.mean()),
                "sigma_min": float(sigma.min()), "sigma_max": float(sigma.max()),
                "shock_mean": float(S.mean()), "shock_min": float(S.min()),
                "shock_max": float(S.max()),
                "cos_pos": F.cosine_similarity(z, tgt, dim=-1)[0].tolist(),
                "y_head": y[0].tolist(),
            }
        return loss, out, S, mets

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        if args.ar:
            starts = torch.randint(0, len(train_ids) - args.n_in - args.ar_m - 1,
                                  (args.batch,), generator=gen)
            xs, ys = [], []
            for j in range(args.ar_m):
                xs.append(torch.stack([train_ids[s + j:s + j + args.n_in] for s in starts]))
                ys.append(torch.stack([train_ids[s + args.n_in + j] for s in starts]))
            x = torch.stack(xs, 1).reshape(-1, args.n_in).to(DEV).long()
            y = torch.stack(ys, 1).reshape(-1, 1).to(DEV).long()
        else:
            starts = torch.randint(0, len(train_ids) - span - 1, (args.batch,), generator=gen)
            x = torch.stack([train_ids[s:s + args.n_in] for s in starts]).to(DEV).long()
            y = torch.stack([train_ids[s + args.n_in:s + args.n_in + args.n_out]
                             for s in starts]).to(DEV).long()
        loss, out, S, mets = loss_step(x, y, prev_shock)
        if not torch.isfinite(loss):
            print(f"s{step} non-finite loss, skip", flush=True)
            continue
        loss.backward()
        groups = {
            "bank": list(model.bank.parameters()),
            "ebank": list(model.ebank.parameters()),
            "dec_a": list(model.dec_a.parameters()),
            "dec_c": list(model.dec_c.parameters()),
            "field": list(model.field.parameters()),
            "tok": [model.tok.weight],
            "out_a": list(model.out_a.parameters()),
            "a_pred": list(model.a_pred.parameters()),
            "flow": list(model.flow.parameters()),
            "role": list(model.role_a.parameters()) + list(model.role_c.parameters()),
            "ea": list(model.ea_proj.parameters()),
        }
        gn_groups = {}
        for gname, gps in groups.items():
            sq = 0.0
            for gp in gps:
                if gp.grad is not None:
                    sq += float((gp.grad.detach() ** 2).sum())
            gn_groups[gname] = sq ** 0.5
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(gn):
            print(f"s{step} non-finite grad, skip", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        with torch.no_grad():
            model.write_memory(out["e_feats"].detach(), model.target_embed(y).detach(),
                               gate=1.0, lr=0.05)
            model.decay_memory(0.98)
        prev_shock = S.detach()
        if args.save_every > 0 and args.save and (step + 1) % args.save_every == 0:
            torch.save({"state": model.state_dict(), "args": vars(args)},
                       args.save.replace(".pt", f".step{step+1}.pt"))
            print(f"s{step} saved periodic ckpt", flush=True)
        if (step + 1) % args.eval_every == 0:
            if args.ar:
                acc_h = eval_ar_acc(model, held_ids, args.n_in, args.ar_m,
                                    args.eval_windows, args.act)
                acc_t = eval_ar_acc(model, train_ids[:200000], args.n_in, args.ar_m,
                                    args.eval_windows, args.act)
            else:
                acc_h = eval_decode_acc(model, held_ids, args.n_in, args.n_out,
                                        args.eval_windows, args.act)
                acc_t = eval_decode_acc(model, train_ids[:200000], args.n_in, args.n_out,
                                        args.eval_windows, args.act)
            print(f"s{step} held_decode_acc={acc_h:.4f} train_decode_acc={acc_t:.4f}", flush=True)
        if step % args.log_every == 0:
            mem = model.memory_norms()
            lr = opt.param_groups[0]["lr"]
            rec = r4({
                "step": step, "loss": float(loss.detach()), **mets,
                "act": args.act, "active": out["active"],
                "mem_mean": float(np.mean(mem)), "mem_max": float(np.max(mem)),
                "grad_clip": float(gn), "lr": float(lr),
                "gn": gn_groups,
                "sec": round(time.time() - t0, 1),
            })
            print("STEP " + json.dumps(rec, ensure_ascii=False), flush=True)
            if args.metrics_jsonl:
                with open(args.metrics_jsonl, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # held 最近邻解码准确率（前 4 个窗口、第一个输出位置）
    if args.ar:
        acc_h = eval_ar_acc(model, held_ids, args.n_in, args.ar_m,
                            args.eval_windows, args.act)
        acc_t = eval_ar_acc(model, train_ids[:200000], args.n_in, args.ar_m,
                            args.eval_windows, args.act)
    else:
        acc_h = eval_decode_acc(model, held_ids, args.n_in, args.n_out,
                                args.eval_windows, args.act)
        acc_t = eval_decode_acc(model, train_ids[:200000], args.n_in, args.n_out,
                                args.eval_windows, args.act)
    print(f"held decode acc={acc_h:.4f} train decode acc={acc_t:.4f} "
          f"total={time.time()-t0:.1f}s", flush=True)
    if args.save:
        torch.save({"state": model.half().state_dict(), "args": vars(args)}, args.save)
        print("saved", args.save)


if __name__ == "__main__":
    main()
