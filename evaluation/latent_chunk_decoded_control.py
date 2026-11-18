"""Test whether a/b latent codes retain task information after decoding."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluation.latent_chunk_ab_thinking import new_model, train_raw
from evaluation.latent_chunk_reconstruction import train_autoencoder
from evaluation.latent_chunk_relations import PHRASE_LENGTH, make_data
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]


def quantize_latent(latent, bits):
    if bits == 32:
        return latent
    if bits == 16:
        return latent.to(torch.float16).to(torch.float32)
    maximum = (1 << (bits - 1)) - 1
    return (latent * maximum).round().clamp(-maximum, maximum) / maximum


@torch.no_grad()
def decode_dataset(source_model, encoder, data, bits, batch_size=256):
    source_model.eval()
    encoder.eval()
    decoded = []

    for start in range(0, data["target"].numel(), batch_size):
        ids = data["ids"][start:start + batch_size]
        a, b = encoder.encode(source_model, ids)
        a = quantize_latent(a, bits)
        b = quantize_latent(b, bits)
        x_ids = encoder.decode(a).argmax(-1)
        y_ids = encoder.decode(b).argmax(-1)
        decoded.append(torch.cat((x_ids, y_ids), dim=-1))

    return {
        "ids": torch.cat(decoded, dim=0),
        "target": data["target"],
        "source": data["source"],
    }


def reconstruction_metrics(predicted_ids, truth_ids):
    x_prediction = predicted_ids[:, :PHRASE_LENGTH]
    y_prediction = predicted_ids[:, PHRASE_LENGTH:]
    x_truth = truth_ids[:, :PHRASE_LENGTH]
    y_truth = truth_ids[:, PHRASE_LENGTH:]
    x_exact = (x_prediction == x_truth).all(-1)
    y_exact = (y_prediction == y_truth).all(-1)
    return {
        "all_16_token_accuracy": float((predicted_ids == truth_ids).float().mean()),
        "both_phrases_exact": float((x_exact & y_exact).float().mean()),
        "query_key_accuracy": float((x_prediction[:, 1] == x_truth[:, 1]).float().mean()),
        "y_edge_symbol_accuracy": float((y_prediction[:, 1:5] == y_truth[:, 1:5]).float().mean()),
        "examples": truth_ids.shape[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4], choices=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bits", type=int, choices=[32, 16, 8, 4], default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="experiments/latent_chunk/decoded_latent_thinking_3seed_rank_800.json",
    )
    cli_args = parser.parse_args()
    args = SimpleNamespace(
        steps=cli_args.steps,
        batch_size=cli_args.batch_size,
        lr=cli_args.lr,
    )
    device = torch.device(cli_args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    report = {
        "task": "two_phrase_key_value_relation_query_with_counterfactual_pairs",
        "condition": f"frozen_a_b_codes_quantized_to_{cli_args.bits}_bits_then_decoded_to_token_ids_before_thinking",
        "method": (
            "Train the same independent phrase autoencoder as the frozen-latent study, quantize "
            "each unit-normalized a/b code with a fixed scale, decode each phrase to argmax token "
            "IDs, then train PBSLM on those IDs using the query loss only. Compare against "
            "raw-token training from the same initial state and minibatch order."
        ),
        "interpretation": (
            "This is an information-retention diagnostic. The decoder is used to build model inputs, "
            "so its parameters and runtime cost are excluded from the stored latent payload."
        ),
        "limitations": (
            "Synthetic task with ordered x/y segmentation and held-out ordered value pairs. The "
            "source token embeddings are frozen random initialization. The decoded control does not "
            "test direct reasoning on continuous latents. Payload size excludes the shared decoder "
            "and assumes bit-packed latent values with no per-vector scale metadata."
        ),
        "config": vars(cli_args),
        "torch": torch.__version__,
        "provenance": capture([
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
            source_model, encoder, reconstruction = train_autoencoder(
                seed, rank, train, test, args, device, initial_state
            )
            decoded_train = decode_dataset(source_model, encoder, train, cli_args.bits)
            decoded_test = decode_dataset(source_model, encoder, test, cli_args.bits)
            decoded_thinking = train_raw(
                seed, decoded_train, decoded_test, args, device, initial_state
            )
            record = {
                "seed": seed,
                "rank": rank,
                "latent_code_bits": cli_args.bits,
                "latent_payload_bytes": 2 * rank * 32 * cli_args.bits // 8,
                "autoencoder_parameters": sum(p.numel() for p in encoder.parameters()),
                "reconstruction": reconstruction,
                "quantized_test_reconstruction": reconstruction_metrics(
                    decoded_test["ids"], test["ids"]
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
