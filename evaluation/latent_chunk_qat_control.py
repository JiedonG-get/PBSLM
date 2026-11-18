"""Test 4-bit quantization-aware training for compact a/b memory codes."""
import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from evaluation.latent_chunk_ab_thinking import new_model, train_raw
from evaluation.latent_chunk_decoded_control import (
    decode_dataset,
    quantize_latent,
    reconstruction_metrics,
)
from evaluation.latent_chunk_reconstruction import SeparateLatentAutoencoder
from evaluation.latent_chunk_relations import PHRASE_LENGTH, make_data
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]


def fake_quantize_latent(latent, bits):
    quantized = quantize_latent(latent, bits)
    return latent + (quantized - latent).detach()


def train_quantized_autoencoder(seed, rank, train, args, device, initial_state):
    source_model = new_model(seed, device)
    source_model.load_state_dict(initial_state)
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)

    encoder = SeparateLatentAutoencoder(source_model.d, rank).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 29)
    started = time.perf_counter()
    final_loss = 0.0
    encoder.train()

    for _ in range(args.steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        ids = train["ids"][indices]
        a, b = encoder.encode(source_model, ids)
        x_logits = encoder.decode(fake_quantize_latent(a, args.bits))
        y_logits = encoder.decode(fake_quantize_latent(b, args.bits))
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
    return source_model, encoder, {
        "final_quantized_batch_loss": final_loss,
        "train_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4], choices=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="experiments/latent_chunk/decoded_latent_int4_qat_3seed_rank_800.json",
    )
    cli_args = parser.parse_args()
    args = SimpleNamespace(
        steps=cli_args.steps,
        batch_size=cli_args.batch_size,
        lr=cli_args.lr,
        bits=cli_args.bits,
    )
    device = torch.device(cli_args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    report = {
        "task": "two_phrase_key_value_relation_query_with_counterfactual_pairs",
        "condition": f"{cli_args.bits}_bit_quantization_aware_a_b_codes_decoded_before_thinking",
        "method": (
            "Train the independent phrase encoder and decoder on train rows with query-independent "
            "full-phrase reconstruction. Apply fixed-scale fake quantization after latent "
            "normalization and before decoding, using an identity straight-through gradient. "
            "Then decode the quantized codes to token IDs and train PBSLM on query loss only."
        ),
        "storage_accounting": (
            "Payload assumes signed bit-packed values with a fixed scale and no per-vector scale "
            "metadata. The decoder is shared model state and excluded from per-example bytes."
        ),
        "limitations": (
            "Synthetic task with ordered x/y segmentation and held-out ordered value pairs. The "
            "source token embeddings are frozen random initialization. The thinking trunk receives "
            "decoder-expanded token IDs, so this does not test direct reasoning on latent vectors."
        ),
        "config": vars(cli_args),
        "torch": torch.__version__,
        "provenance": capture([
            "evaluation/latent_chunk_qat_control.py",
            "evaluation/latent_chunk_decoded_control.py",
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
        train, test = make_data(seed, device)
        initial_model = new_model(seed, device)
        initial_state = copy.deepcopy(initial_model.state_dict())
        del initial_model
        raw_baseline = train_raw(seed, train, test, args, device, initial_state)

        for rank in cli_args.ranks:
            source_model, encoder, training = train_quantized_autoencoder(
                seed, rank, train, args, device, initial_state
            )
            quantized_train = decode_dataset(source_model, encoder, train, cli_args.bits)
            quantized_test = decode_dataset(source_model, encoder, test, cli_args.bits)
            decoded_thinking = train_raw(
                seed, quantized_train, quantized_test, args, device, initial_state
            )
            record = {
                "seed": seed,
                "rank": rank,
                "latent_code_bits": cli_args.bits,
                "latent_payload_bytes": 2 * rank * 32 * cli_args.bits // 8,
                "autoencoder_parameters": sum(p.numel() for p in encoder.parameters()),
                "quantization_aware_training": training,
                "quantized_train_reconstruction": reconstruction_metrics(
                    quantized_train["ids"], train["ids"]
                ),
                "quantized_test_reconstruction": reconstruction_metrics(
                    quantized_test["ids"], test["ids"]
                ),
                "decoded_latent_thinking": decoded_thinking,
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
