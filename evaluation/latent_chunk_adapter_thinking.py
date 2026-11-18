"""Test a residual input adapter for direct a/b latent reasoning."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.latent_chunk_ab_thinking import class_logits, new_model, train_raw
from evaluation.latent_chunk_reconstruction import train_autoencoder, train_thinking_trunk
from evaluation.latent_chunk_relations import CLASSES, PHRASE_LENGTH, make_data
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY_OFFSET = 3


def confirmatory_split(seed, device):
    train, _ = make_data(seed, device)
    held_out_pairs = {(value, (value + CONFIRMATORY_OFFSET) % CLASSES) for value in range(CLASSES)}
    value_pairs = train["ids"][0::2, PHRASE_LENGTH + 2:PHRASE_LENGTH + 5:2]
    selected_pairs = torch.tensor(
        [tuple(pair) in held_out_pairs for pair in value_pairs.cpu().tolist()],
        dtype=torch.bool,
        device=device,
    )
    test_rows = selected_pairs.repeat_interleave(2)
    test = {key: value[test_rows] for key, value in train.items()}
    train = {key: value[~test_rows] for key, value in train.items()}

    train_ids = {tuple(row) for row in train["ids"].cpu().tolist()}
    test_ids = {tuple(row) for row in test["ids"].cpu().tolist()}
    assert train_ids.isdisjoint(test_ids)
    assert all(tuple(pair) in held_out_pairs for pair in value_pairs[selected_pairs].cpu().tolist())
    assert set(train["target"].cpu().tolist()) == set(range(CLASSES))
    return train, test, sorted(held_out_pairs)


class ResidualLatentAdapter(nn.Module):
    """Map each latent vector toward the trunk input space before adding tags."""

    def __init__(self, rank, d):
        super().__init__()
        self.first = nn.Linear(d, 2 * d)
        self.second = nn.Linear(2 * d, d)
        nn.init.zeros_(self.second.weight)
        nn.init.zeros_(self.second.bias)
        self.role = nn.Parameter(torch.randn(2, rank, d) * 0.02)
        self.position = nn.Parameter(torch.randn(2, rank, d) * 0.02)

    def apply(self, a, b):
        latent = torch.cat((a, b), dim=1)
        latent = latent + self.second(F.gelu(self.first(latent)))
        tags = torch.cat((
            self.role[0] + self.position[0],
            self.role[1] + self.position[1],
        ), dim=0)
        return latent + tags


@torch.no_grad()
def evaluate_adapter(model, source_model, encoder, adapter, data, batch_size=256):
    model.eval()
    source_model.eval()
    encoder.eval()
    adapter.eval()
    predictions = []
    for start in range(0, data["target"].numel(), batch_size):
        sl = slice(start, min(start + batch_size, data["target"].numel()))
        ids = data["ids"][sl]
        a, b = encoder.encode(source_model, ids)
        output = model(None, 1, inputs_embeds=adapter.apply(a, b), sample_noise=False)
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


def train_adapter(seed, rank, source_model, encoder, train, test, args, device, initial_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_state)
    adapter = ResidualLatentAdapter(rank, model.d).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad] + list(adapter.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    torch.manual_seed(seed * 1009 + 7)
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 19)
    model.train()
    adapter.train()

    for _ in range(args.steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        ids = train["ids"][indices]
        with torch.no_grad():
            a, b = encoder.encode(source_model, ids)
        output = model(None, 1, inputs_embeds=adapter.apply(a, b), sample_noise=False)
        objective = F.cross_entropy(class_logits(model, output), train["target"][indices])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    return {
        "test": evaluate_adapter(model, source_model, encoder, adapter, test),
        "train": evaluate_adapter(model, source_model, encoder, adapter, train),
        "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--ranks", nargs="+", type=int, default=[2, 4], choices=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="experiments/latent_chunk/direct_adapter_thinking_3seed_rank_800.json",
    )
    cli_args = parser.parse_args()
    args = SimpleNamespace(steps=cli_args.steps, batch_size=cli_args.batch_size, lr=cli_args.lr)
    device = torch.device(cli_args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    report = {
        "task": "two_phrase_key_value_relation_query_with_counterfactual_pairs",
        "condition": "frozen_reconstruction_latents_with_zero_initialized_residual_input_adapter",
        "method": (
            "Move the offset-3 ordered value-pair family out of the standard training rows before "
            "training either encoder or trunk. Train the phrase autoencoder on remaining rows, "
            "freeze its codes, then train a zero-initialized residual MLP adapter and PBSLM with "
            "query loss only. Compare with the existing direct latent trunk and raw-token trunk."
        ),
        "confirmatory_split": {
            "held_out_value_pairs": sorted(
                (value, (value + CONFIRMATORY_OFFSET) % CLASSES) for value in range(CLASSES)
            ),
            "source": "rows removed from the standard training pool before encoder and task training",
            "previous_standard_test_split_used": False,
        },
        "limitations": (
            "Synthetic relation task with explicitly ordered x/y spans and eight held-out value pairs. "
            "This tests a single predeclared adapter on one new split, not natural-language or online "
            "memory performance. Latent sequence length remains rank-dependent."
        ),
        "config": vars(cli_args),
        "torch": torch.__version__,
        "provenance": capture([
            "evaluation/latent_chunk_adapter_thinking.py",
            "evaluation/latent_chunk_reconstruction.py",
            "evaluation/latent_chunk_ab_thinking.py",
            "evaluation/latent_chunk_relations.py",
            "evaluation/provenance.py",
            "models/model.py",
            "models/model_v2.py",
        ]),
        "runs": [],
    }

    for seed in cli_args.seeds:
        train, test, held_out_pairs = confirmatory_split(seed, device)
        initial_model = new_model(seed, device)
        initial_state = copy.deepcopy(initial_model.state_dict())
        del initial_model
        raw_baseline = train_raw(seed, train, test, args, device, initial_state)

        for rank in cli_args.ranks:
            source_model, encoder, reconstruction = train_autoencoder(
                seed, rank, train, test, args, device, initial_state
            )
            frozen_latent = train_thinking_trunk(
                seed, rank, source_model, encoder, train, test, args, device, initial_state
            )
            adapted_latent = train_adapter(
                seed, rank, source_model, encoder, train, test, args, device, initial_state
            )
            record = {
                "seed": seed,
                "rank": rank,
                "held_out_value_pairs": held_out_pairs,
                "train_examples": train["target"].numel(),
                "test_examples": test["target"].numel(),
                "latent_payload_bytes_fp32": 2 * rank * 32 * 4,
                "autoencoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
                "reconstruction": reconstruction,
                "frozen_direct_latent_thinking": frozen_latent,
                "adapted_direct_latent_thinking": adapted_latent,
                "raw_token_baseline": raw_baseline,
            }
            report["runs"].append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            del source_model, encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()

    path = ROOT / cli_args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
