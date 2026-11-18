"""Measure a/b phrase reconstruction and frozen-latent PBSLM reasoning."""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.latent_chunk_ab_thinking import class_logits, new_model, train_raw
from evaluation.latent_chunk_relations import (
    CLASSES,
    PHRASE_LENGTH,
    TARGET_START,
    VOCAB,
    make_data,
)
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]
PHRASE_VOCAB = CLASSES + 2


class SeparateLatentAutoencoder(nn.Module):
    """Compress each phrase independently and decode all of its token IDs."""

    def __init__(self, d, rank):
        super().__init__()
        self.d = d
        self.rank = rank
        self.encoder = nn.Sequential(
            nn.Linear(PHRASE_LENGTH * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, rank * d),
        )
        self.decoder = nn.Sequential(
            nn.Linear(rank * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, PHRASE_LENGTH * PHRASE_VOCAB),
        )

    def encode(self, source_model, ids):
        with torch.no_grad():
            token_vectors = source_model.token_emb(ids)
        x_tokens = token_vectors[:, :PHRASE_LENGTH]
        y_tokens = token_vectors[:, PHRASE_LENGTH:]
        a = self.encoder(x_tokens.flatten(1)).view(-1, self.rank, self.d)
        b = self.encoder(y_tokens.flatten(1)).view(-1, self.rank, self.d)
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return a, b

    def decode(self, latent):
        logits = self.decoder(latent.flatten(1))
        return logits.view(-1, PHRASE_LENGTH, PHRASE_VOCAB)


class LatentRoleTags(nn.Module):
    """Trainable input-role markers used only by the downstream thinking trunk."""

    def __init__(self, rank, d):
        super().__init__()
        self.role = nn.Parameter(torch.randn(2, rank, d) * 0.02)
        self.position = nn.Parameter(torch.randn(2, rank, d) * 0.02)

    def apply(self, a, b):
        return torch.cat((
            a + self.role[0] + self.position[0],
            b + self.role[1] + self.position[1],
        ), dim=1)


def reconstruction_metrics(encoder, source_model, data, batch_size=256):
    encoder.eval()
    source_model.eval()
    x_correct = torch.zeros(PHRASE_LENGTH, device=data["ids"].device)
    y_correct = torch.zeros(PHRASE_LENGTH, device=data["ids"].device)
    x_exact = 0
    y_exact = 0
    joint_exact = 0
    total = data["target"].numel()

    with torch.no_grad():
        for start in range(0, total, batch_size):
            sl = slice(start, min(start + batch_size, total))
            ids = data["ids"][sl]
            a, b = encoder.encode(source_model, ids)
            x_prediction = encoder.decode(a).argmax(-1)
            y_prediction = encoder.decode(b).argmax(-1)
            x_truth = ids[:, :PHRASE_LENGTH]
            y_truth = ids[:, PHRASE_LENGTH:]
            x_correct += (x_prediction == x_truth).sum(0)
            y_correct += (y_prediction == y_truth).sum(0)
            x_match = (x_prediction == x_truth).all(-1)
            y_match = (y_prediction == y_truth).all(-1)
            x_exact += int(x_match.sum())
            y_exact += int(y_match.sum())
            joint_exact += int((x_match & y_match).sum())

    x_accuracy = (x_correct / total).tolist()
    y_accuracy = (y_correct / total).tolist()
    variable_positions = slice(1, None)
    edge_positions = slice(1, 5)
    return {
        "x_token_accuracy": sum(x_accuracy) / PHRASE_LENGTH,
        "y_token_accuracy": sum(y_accuracy) / PHRASE_LENGTH,
        "all_16_token_accuracy": (sum(x_accuracy) + sum(y_accuracy)) / (2 * PHRASE_LENGTH),
        "x_variable_token_accuracy": sum(x_accuracy[1:]) / (PHRASE_LENGTH - 1),
        "y_variable_token_accuracy": sum(y_accuracy[1:]) / (PHRASE_LENGTH - 1),
        "query_key_accuracy": x_accuracy[1],
        "y_edge_symbol_accuracy": sum(y_accuracy[edge_positions]) / 4,
        "x_filler_accuracy": sum(x_accuracy[2:]) / (PHRASE_LENGTH - 2),
        "y_filler_accuracy": sum(y_accuracy[5:]) / 3,
        "x_phrase_exact": x_exact / total,
        "y_phrase_exact": y_exact / total,
        "both_phrases_exact": joint_exact / total,
        "position_accuracy": {"x": x_accuracy, "y": y_accuracy},
        "examples": total,
    }


def train_autoencoder(seed, rank, train, test, args, device, initial_state):
    source_model = new_model(seed, device)
    source_model.load_state_dict(initial_state)
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)

    encoder = SeparateLatentAutoencoder(source_model.d, rank).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    encoder.train()
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 29)
    started = time.perf_counter()
    final_loss = 0.0

    for _ in range(args.steps):
        indices = torch.randint(train["target"].numel(), (args.batch_size,), generator=rng, device=device)
        ids = train["ids"][indices]
        a, b = encoder.encode(source_model, ids)
        x_logits = encoder.decode(a)
        y_logits = encoder.decode(b)
        x_truth = ids[:, :PHRASE_LENGTH]
        y_truth = ids[:, PHRASE_LENGTH:]
        objective = (
            F.cross_entropy(x_logits.flatten(0, 1), x_truth.flatten())
            + F.cross_entropy(y_logits.flatten(0, 1), y_truth.flatten())
        ) / 2
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        final_loss = float(objective.detach())

    encoder.eval()
    result = {
        "train_reconstruction": reconstruction_metrics(encoder, source_model, train),
        "test_reconstruction": reconstruction_metrics(encoder, source_model, test),
        "final_batch_loss": final_loss,
        "train_seconds": time.perf_counter() - started,
    }
    return source_model, encoder, result


def evaluate_latent_task(model, source_model, encoder, tags, data, batch_size=256):
    model.eval()
    source_model.eval()
    encoder.eval()
    tags.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, data["target"].numel(), batch_size):
            sl = slice(start, min(start + batch_size, data["target"].numel()))
            ids = data["ids"][sl]
            a, b = encoder.encode(source_model, ids)
            embeds = tags.apply(a, b)
            output = model(None, 1, inputs_embeds=embeds, sample_noise=False)
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


def train_thinking_trunk(seed, rank, source_model, encoder, train, test, args, device, initial_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_state)
    tags = LatentRoleTags(rank, model.d).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(tags.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    torch.manual_seed(seed * 1009 + 7)
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 19)
    model.train()
    tags.train()

    for _ in range(args.steps):
        indices = torch.randint(train["target"].numel(), (args.batch_size,), generator=rng, device=device)
        ids = train["ids"][indices]
        with torch.no_grad():
            a, b = encoder.encode(source_model, ids)
        output = model(None, 1, inputs_embeds=tags.apply(a, b), sample_noise=False)
        objective = F.cross_entropy(class_logits(model, output), train["target"][indices])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] + list(tags.parameters()), 1.0
        )
        optimizer.step()

    return {
        "test": evaluate_latent_task(model, source_model, encoder, tags, test),
        "train": evaluate_latent_task(model, source_model, encoder, tags, train),
        "trainable_trunk_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_role_tag_parameters": sum(p.numel() for p in tags.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4], choices=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="experiments/latent_chunk/reconstruction_thinking_3seed_rank_800.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    report = {
        "task": "two_phrase_key_value_relation_query_with_counterfactual_pairs",
        "condition": "full_phrase_autoencoding_then_frozen_a_b_latent_reasoning",
        "method": (
            "Train the independent shared span encoder on full x/y token reconstruction using train rows only. "
            "Freeze its encoder and train PBSLMv2 on a(x), b(y) with query-target loss only."
        ),
        "limitations": (
            "Synthetic relation task with ordered x/y segmentation. Reconstruction is evaluated on held-out rows "
            "including value pairs absent from training. The encoder reads a frozen randomly initialized token "
            "embedding table. The downstream comparison is not parameter- or FLOP-matched."
        ),
        "config": vars(args),
        "torch": torch.__version__,
        "provenance": capture([
            "evaluation/latent_chunk_reconstruction.py",
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
        raw_baseline = train_raw(seed, train, test, args, device, initial_state)
        for rank in args.ranks:
            source_model, encoder, reconstruction = train_autoencoder(
                seed, rank, train, test, args, device, initial_state
            )
            thinking = train_thinking_trunk(
                seed, rank, source_model, encoder, train, test, args, device, initial_state
            )
            record = {
                "seed": seed,
                "rank": rank,
                "latent_payload_bytes_fp32": 2 * rank * 32 * 4,
                "autoencoder_parameters": sum(p.numel() for p in encoder.parameters()),
                "reconstruction": reconstruction,
                "frozen_latent_thinking": thinking,
                "raw_token_baseline": raw_baseline,
            }
            report["runs"].append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            del source_model, encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()

    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
