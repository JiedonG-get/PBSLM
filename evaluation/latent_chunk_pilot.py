"""Controlled pilot for multi-token a/b/c/d latents as PBSLMv2 inputs.

The task is synthetic relational recall, not a natural-language benchmark.
Each side is an eight-token phrase. x queries one of two key-value edges in y.
Counterfactual test rows share y while changing x, and held-out edge-value pairs
test composition. c and d are computed only from a and b.

Run from the repository root with:
  C:/Python313/python.exe -m evaluation.latent_chunk_pilot --seeds 0 1 2
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.latent_chunk_relations import CLASSES, PHRASE_LENGTH, TARGET_START, VOCAB, make_data
from evaluation.provenance import capture
from evaluation.latent_chunk_state import evaluate_state
from models.model_v2 import PBSLMv2

ROOT = Path(__file__).resolve().parents[1]


class ChunkLatents(nn.Module):
    def __init__(self, d, rank):
        super().__init__()
        self.d = d
        self.rank = rank
        self.span = nn.Sequential(
            nn.Linear(PHRASE_LENGTH * d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, rank * d),
        )
        self.joint = nn.Sequential(
            nn.Linear(2 * d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, rank * d),
        )
        self.role = nn.Parameter(torch.randn(4, rank, d) * 0.02)
        self.position = nn.Parameter(torch.randn(4, rank, d) * 0.02)
        self.x_reconstruct = nn.Linear(d, CLASSES)
        self.y_reconstruct = nn.Linear(d, 4 * CLASSES)
        self.joint_probe = nn.Linear(d, CLASSES)

    def encode(self, model, ids):
        token_vectors = model.token_emb(ids)
        x_tokens, y_tokens = token_vectors[:, :PHRASE_LENGTH], token_vectors[:, PHRASE_LENGTH:]
        a = self.span(x_tokens.flatten(1)).view(-1, self.rank, self.d)
        b = self.span(y_tokens.flatten(1)).view(-1, self.rank, self.d)
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)

        def pair_code(left, right):
            features = torch.cat((left.mean(1), right.mean(1)), dim=-1)
            return F.normalize(self.joint(features).view(-1, self.rank, self.d), dim=-1)

        c = pair_code(a, b)
        d = pair_code(b, a)
        groups = (a, b, c, d)
        tagged = [group + self.role[i] + self.position[i] for i, group in enumerate(groups)]
        return groups, tagged

    def auxiliaries(self, groups):
        a, b, c, d = groups
        c_mean, d_mean = c.mean(1), d.mean(1)
        alignment = 1 - F.cosine_similarity(c_mean, d_mean, dim=-1).mean()
        return alignment


def new_model(seed, device):
    # Keep the PBSLM trunk initialization fixed across latent ranks.
    torch.manual_seed(seed * 1009)
    model = PBSLMv2(
        vocab=VOCAB, k=4, depth=1, d=32, heads=4, m_max=1,
        act_x=0.5, e_n=2, e_depth=1, refine=1, field_scale=0,
        trainable_target=False, use_mem=False, use_flow=False,
        dense=True, use_sample_masks=False,
    ).to(device)
    model.eval()
    with torch.no_grad():
        model.ctx_a.weight.zero_()
        model.ctx_a.bias.zero_()
        model.ctx_c.weight.zero_()
        model.ctx_c.bias.zero_()
    for parameter in (*model.ctx_a.parameters(), *model.ctx_c.parameters()):
        parameter.requires_grad_(False)
    return model


def class_logits(model, output):
    class_codes = model.code[TARGET_START:TARGET_START + CLASSES].float()
    return output["z_a"][:, 0].float() @ class_codes.t() / 0.1


def latent_inputs(encoder, model, ids, mode, mismatch_index=None):
    groups, tagged = encoder.encode(model, ids)
    if mode == "full":
        selected = tagged
    elif mode == "a_only":
        selected = tagged[:1]
    elif mode == "b_only":
        selected = tagged[1:2]
    elif mode == "ab":
        selected = tagged[:2]
    elif mode == "cd":
        selected = tagged[2:]
    elif mode == "zero_cd":
        selected = [*tagged[:2], torch.zeros_like(tagged[2]), torch.zeros_like(tagged[3])]
    elif mode == "mismatch_cd":
        selected = [*tagged[:2], tagged[2][mismatch_index], tagged[3][mismatch_index]]
    else:
        raise ValueError(mode)
    return torch.cat(selected, dim=1), groups


def predict_latent(model, encoder, batch, mode, batch_index=None):
    mismatch = None
    if mode == "mismatch_cd":
        mismatch = batch_index ^ 1
    embeds, groups = latent_inputs(encoder, model, batch["ids"], mode, mismatch)
    out = model(None, 1, inputs_embeds=embeds, sample_noise=False)
    return class_logits(model, out), groups


def evaluate_latent(model, encoder, data, batch_size=256):
    model.eval()
    encoder.eval()
    names = ("full", "a_only", "b_only", "ab", "cd", "zero_cd", "mismatch_cd")
    correct = {name: 0 for name in names}
    joint_probe_correct = 0
    x_key_correct = 0
    y_edge_correct = 0
    full_predictions = []
    total = data["target"].numel()
    with torch.no_grad():
        for start in range(0, total, batch_size):
            sl = slice(start, min(start + batch_size, total))
            batch = {key: value[sl] for key, value in data.items()}
            local_index = torch.arange(batch["target"].numel(), device=batch["target"].device)
            for name in names:
                logits, groups = predict_latent(model, encoder, batch, name, local_index)
                correct[name] += int((logits.argmax(-1) == batch["target"]).sum())
                if name == "full":
                    full_predictions.extend(logits.argmax(-1).tolist())
            groups, _ = encoder.encode(model, batch["ids"])
            x_key_correct += int((encoder.x_reconstruct(groups[0].mean(1)).argmax(-1) ==
                                  batch["ids"][:, 1]).sum())
            y_edges = encoder.y_reconstruct(groups[1].mean(1)).view(-1, 4, CLASSES)
            y_edge_correct += int((y_edges.argmax(-1) ==
                                   batch["ids"][:, PHRASE_LENGTH + 1:PHRASE_LENGTH + 5]).sum())
            joint_probe_correct += int((encoder.joint_probe((groups[2].mean(1) + groups[3].mean(1)) / 2).argmax(-1)
                                        == batch["target"]).sum())
    predictions = torch.tensor(full_predictions, dtype=torch.long, device=data["target"].device)
    pairwise_exact = float((predictions[0::2].eq(data["target"][0::2]) &
                            predictions[1::2].eq(data["target"][1::2])).float().mean())
    return {
        "accuracy": {name: value / total for name, value in correct.items()},
        "counterfactual_pair_exact": pairwise_exact,
        "content_reconstruction_accuracy": {"a_query_key": x_key_correct / total,
                                            "b_edge_symbols": y_edge_correct / (4 * total)},
        "cd_joint_probe_accuracy": joint_probe_correct / total,
        "examples": total,
    }


def train_latent(seed, rank, mode, train, test, args, device, initial_model_state=None):
    model = new_model(seed, device)
    encoder = ChunkLatents(model.d, rank).to(device)
    if initial_model_state is not None:
        model.load_state_dict(initial_model_state)
    torch.manual_seed(seed * 1009 + 7)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(encoder.parameters()), lr=args.lr, weight_decay=1e-4
    )
    model.train()
    encoder.train()
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 19)
    started = time.perf_counter()
    steps = args.steps
    final_losses = {}
    for step in range(steps):
        indices = torch.randint(train["target"].numel(), (args.batch_size,), generator=rng, device=device)
        batch = {key: value[indices] for key, value in train.items()}
        embeds, groups = latent_inputs(encoder, model, batch["ids"], mode)
        output = model(None, 1, inputs_embeds=embeds, sample_noise=False)
        logits = class_logits(model, output)
        a, b, c, d = groups
        phrase_loss = F.cross_entropy(encoder.x_reconstruct(a.mean(1)), batch["ids"][:, 1])
        y_edge_targets = batch["ids"][:, PHRASE_LENGTH + 1:PHRASE_LENGTH + 5]
        phrase_loss += F.cross_entropy(
            encoder.y_reconstruct(b.mean(1)).view(-1, 4, CLASSES).flatten(0, 1),
            y_edge_targets.flatten(),
        )
        joint_target = F.cross_entropy(
            encoder.joint_probe((c.mean(1) + d.mean(1)) / 2), batch["target"]
        )
        alignment = encoder.auxiliaries(groups)
        objective = F.cross_entropy(logits, batch["target"])
        objective = objective + 0.25 * phrase_loss + 0.35 * joint_target + 0.15 * alignment
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(encoder.parameters()), 1.0)
        optimizer.step()
        if step >= steps - min(50, steps):
            final_losses = {
                "total": float(objective.detach()),
                "main": float(F.cross_entropy(logits.detach(), batch["target"])),
                "phrase": float(phrase_loss.detach()),
                "joint_probe": float(joint_target.detach()),
                "alignment": float(alignment.detach()),
            }
    model.eval()
    encoder.eval()
    scores = evaluate_latent(model, encoder, test)
    scores["train_evaluation"] = evaluate_latent(model, encoder, train)
    scores["final_batch_losses"] = final_losses
    scores["train_seconds"] = time.perf_counter() - started
    scores["mode"] = mode
    return model, encoder, scores


def train_raw(seed, train, test, args, device, initial_model_state=None):
    model = new_model(seed, device)
    if initial_model_state is not None:
        model.load_state_dict(initial_model_state)
    torch.manual_seed(seed * 1009 + 7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 19)
    started = time.perf_counter()
    for _ in range(args.steps):
        indices = torch.randint(train["target"].numel(), (args.batch_size,), generator=rng, device=device)
        batch = {key: value[indices] for key, value in train.items()}
        output = model(batch["ids"], 1, sample_noise=False)
        objective = F.cross_entropy(class_logits(model, output), batch["target"])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, test["target"].numel(), 256):
            sl = slice(start, min(start + 256, test["target"].numel()))
            output = model(test["ids"][sl], 1, sample_noise=False)
            correct += int((class_logits(model, output).argmax(-1) == test["target"][sl]).sum())
    return {"accuracy": correct / test["target"].numel(), "train_seconds": time.perf_counter() - started}


@torch.no_grad()
def evaluate_latent_state(model, encoder, data, seed):
    model.eval()
    encoder.eval()
    groups, tagged = encoder.encode(model, data["ids"])
    payload = torch.stack(tagged, dim=1)

    def predict(stored):
        embeds = stored.flatten(1, 2)
        output = model(None, 1, inputs_embeds=embeds, sample_noise=False)
        return class_logits(model, output)

    return evaluate_state(payload, data["target"], predict, seed=seed)


def storage_report(d=32, dtype_bytes=4, tokens=2 * PHRASE_LENGTH):
    return {
        "input_token_ids_bytes_int32": tokens * 4,
        "input_token_ids_bytes_torch_long": tokens * 8,
        "input_embedding_bytes": tokens * d * dtype_bytes,
        "latent_payload_bytes": {
            str(rank): 4 * rank * d * dtype_bytes for rank in (1, 2, 4)
        },
        "two_sided_ab_payload_bytes": {
            str(rank): 2 * rank * d * dtype_bytes for rank in (1, 2, 4)
        },
        "input_kv_bytes_by_layers": {
            str(layers): 2 * layers * tokens * d * dtype_bytes for layers in (1, 4, 12)
        },
        "note": "Float32 payload estimates. Excludes encoder/decoder parameters, metadata, allocator overhead, and fixed shared state. KV uses 2*K/V tensors per layer and token. Latent payload is not an implemented online memory state.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4], choices=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="experiments/latent_chunk/pilot.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    report = {
        "task": "two_phrase_key_value_relation_query_with_counterfactual_pairs",
            "limitation": "Synthetic relational task. Tests continuous latent inputs through PBSLMv2's shared encoder and decoder with its direct context bypass zeroed. Does not test online fast-state writing during training or natural-language semantics. Model initialization and minibatch order are shared across latent ranks and training modes within each seed. Latent models add encoder parameters and auxiliary supervision, so this is not parameter- or FLOP-matched.",
        "config": vars(args),
        "torch": torch.__version__,
        "provenance": capture([
            "evaluation/latent_chunk_pilot.py", "evaluation/latent_chunk_relations.py",
            "evaluation/latent_chunk_state.py", "evaluation/provenance.py",
            "models/model.py", "models/model_v2.py",
        ]),
        "storage": storage_report(),
        "runs": [],
    }
    for seed in args.seeds:
        train, test = make_data(seed, device)
        model_seed = seed * 1009
        torch.manual_seed(model_seed)
        initial_model = new_model(seed, device)
        initial_state = copy.deepcopy(initial_model.state_dict())
        del initial_model
        for rank in args.ranks:
            full_model, full_encoder, full_result = train_latent(
                seed, rank, "full", train, test, args, device, initial_state
            )
            state_result = evaluate_latent_state(full_model, full_encoder, test, seed + 100)
            _, _, ab_result = train_latent(
                seed, rank, "ab", train, test, args, device, initial_state
            )
            raw_result = train_raw(seed, train, test, args, device, initial_state)
            record = {
                "seed": seed,
                "rank": rank,
                "test_examples": test["target"].numel(),
                "trainable_model_parameters": sum(
                    parameter.numel() for parameter in full_model.parameters()
                    if parameter.requires_grad
                ),
                "latent_encoder_parameters": sum(
                    parameter.numel() for parameter in full_encoder.parameters()
                    if parameter.requires_grad
                ),
                "full": full_result,
                "ab_only_trained": ab_result,
                "raw_token_baseline": raw_result,
                "fixed_state_retrieval": state_result,
            }
            report["runs"].append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            del full_model, full_encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
