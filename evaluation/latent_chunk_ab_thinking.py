"""Test whether PBSLM can compose separate x/y latent inputs without c/d codes.

The model receives only a(x) and b(y). Training uses only the relation-query
target. No pair-combination latent, reconstruction probe, or alignment loss is
constructed.

Run from the repository root with:
  C:/Python313/python.exe -m evaluation.latent_chunk_ab_thinking --seeds 0 1 2
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.latent_chunk_relations import (
    CLASSES,
    PHRASE_LENGTH,
    TARGET_START,
    VOCAB,
    make_data,
)
from evaluation.provenance import capture
from models.model_v2 import PBSLMv2

ROOT = Path(__file__).resolve().parents[1]


class SeparateLatents(nn.Module):
    """Encode x and y independently into tagged rank-r vector sequences."""

    def __init__(self, d, rank):
        super().__init__()
        self.d = d
        self.rank = rank
        self.span = nn.Sequential(
            nn.Linear(PHRASE_LENGTH * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, rank * d),
        )
        self.role = nn.Parameter(torch.randn(2, rank, d) * 0.02)
        self.position = nn.Parameter(torch.randn(2, rank, d) * 0.02)

    def encode(self, model, ids):
        token_vectors = model.token_emb(ids)
        x_tokens = token_vectors[:, :PHRASE_LENGTH]
        y_tokens = token_vectors[:, PHRASE_LENGTH:]
        a = F.normalize(self.span(x_tokens.flatten(1)).view(-1, self.rank, self.d), dim=-1)
        b = F.normalize(self.span(y_tokens.flatten(1)).view(-1, self.rank, self.d), dim=-1)
        tagged = [
            a + self.role[0] + self.position[0],
            b + self.role[1] + self.position[1],
        ]
        return a, b, tagged


def new_model(seed, device):
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


@torch.no_grad()
def evaluate_latent(model, encoder, data, batch_size=256):
    model.eval()
    encoder.eval()
    total = data["target"].numel()
    correct = 0
    predictions = []
    for start in range(0, total, batch_size):
        sl = slice(start, min(start + batch_size, total))
        batch_ids = data["ids"][sl]
        targets = data["target"][sl]
        _, _, tagged = encoder.encode(model, batch_ids)
        embeds = torch.cat(tagged, dim=1)
        output = model(None, 1, inputs_embeds=embeds, sample_noise=False)
        logits = class_logits(model, output)
        predicted = logits.argmax(-1)
        correct += int((predicted == targets).sum())
        predictions.extend(predicted.tolist())

    predictions = torch.tensor(predictions, device=data["target"].device)
    pair_exact = (
        predictions[0::2].eq(data["target"][0::2])
        & predictions[1::2].eq(data["target"][1::2])
    ).float().mean()
    return {
        "accuracy": correct / total,
        "counterfactual_pair_exact": float(pair_exact),
        "examples": total,
    }


def train_latent(seed, rank, train, test, args, device, initial_model_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_model_state)
    encoder = SeparateLatents(model.d, rank).to(device)
    torch.manual_seed(seed * 1009 + 7)
    trainable = [p for p in (*model.parameters(), *encoder.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    model.train()
    encoder.train()
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 19)
    started = time.perf_counter()
    final_losses = {}

    for step in range(args.steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        batch = {key: value[indices] for key, value in train.items()}
        a, b, tagged = encoder.encode(model, batch["ids"])
        output = model(None, 1, inputs_embeds=torch.cat(tagged, dim=1), sample_noise=False)
        target_loss = F.cross_entropy(class_logits(model, output), batch["target"])
        objective = target_loss
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step >= args.steps - min(50, args.steps):
            final_losses = {
                "total": float(objective.detach()),
                "target": float(target_loss.detach()),
            }

    test_result = evaluate_latent(model, encoder, test)
    test_result["train_evaluation"] = evaluate_latent(model, encoder, train)
    test_result["final_batch_losses"] = final_losses
    test_result["train_seconds"] = time.perf_counter() - started
    return model, encoder, test_result


def train_raw(seed, train, test, args, device, initial_model_state):
    model = new_model(seed, device)
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

    test_result = evaluate_raw(model, test)
    test_result["train_evaluation"] = evaluate_raw(model, train)
    test_result["train_seconds"] = time.perf_counter() - started
    return test_result


@torch.no_grad()
def evaluate_raw(model, data, batch_size=256):
    model.eval()
    predictions = []
    for start in range(0, data["target"].numel(), batch_size):
        sl = slice(start, min(start + batch_size, data["target"].numel()))
        output = model(data["ids"][sl], 1, sample_noise=False)
        predictions.extend(class_logits(model, output).argmax(-1).tolist())
    predictions = torch.tensor(predictions, device=data["target"].device)
    pair_exact = (
        predictions[0::2].eq(data["target"][0::2])
        & predictions[1::2].eq(data["target"][1::2])
    ).float().mean()
    return {
        "accuracy": float((predictions == data["target"]).float().mean()),
        "counterfactual_pair_exact": float(pair_exact),
        "examples": data["target"].numel(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4], choices=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="experiments/latent_chunk/pure_ab_thinking_3seed_rank_800.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    report = {
        "task": "two_phrase_key_value_relation_query_with_counterfactual_pairs",
        "condition": "separate_a_x_and_b_y_latents_only",
        "limitation": (
            "Synthetic relation task. The trunk receives only a(x) and b(y), trained with the "
            "query target loss alone. No c/d codes, probes, reconstruction heads, or alignment "
            "losses are used. Parameter and FLOP counts differ from the raw-token baseline."
        ),
        "config": vars(args),
        "torch": torch.__version__,
        "provenance": capture([
            "evaluation/latent_chunk_ab_thinking.py",
            "evaluation/latent_chunk_relations.py",
            "evaluation/provenance.py",
            "models/model.py",
            "models/model_v2.py",
        ]),
        "runs": [],
    }

    for seed in args.seeds:
        train, test = make_data(seed, device)
        initial_model = new_model(seed, device)
        initial_state = copy.deepcopy(initial_model.state_dict())
        del initial_model
        for rank in args.ranks:
            latent_model, encoder, latent_result = train_latent(
                seed, rank, train, test, args, device, initial_state
            )
            raw_result = train_raw(seed, train, test, args, device, initial_state)
            record = {
                "seed": seed,
                "rank": rank,
                "test_examples": test["target"].numel(),
                "latent_input_vectors": 2 * rank,
                "latent_payload_bytes_fp32": 2 * rank * 32 * 4,
                "trainable_model_parameters": sum(
                    p.numel() for p in latent_model.parameters() if p.requires_grad
                ),
                "latent_encoder_parameters": sum(
                    p.numel() for p in encoder.parameters() if p.requires_grad
                ),
                "pure_ab_thinking": latent_result,
                "raw_token_baseline": raw_result,
            }
            report["runs"].append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            del latent_model, encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()

    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
