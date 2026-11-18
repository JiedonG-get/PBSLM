"""Separate data-sampling variance from model-seed variance for latent readers.

The offset-4 dataset is sampled once with data_seed and then reused for every
model seed. A rank-4 phrase autoencoder is trained once per model seed. Raw,
direct-int4, soft-reader, and argmax-decoded task models are then trained from
the same initial PBSLM state at each requested task-training budget.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluation import latent_chunk_soft_reader_confirmatory as reader
from evaluation.latent_chunk_ab_thinking import new_model, train_raw
from evaluation.latent_chunk_reconstruction import train_autoencoder
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]
RANK = 4
AUTOENCODER_STEPS = 800
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def run_task_conditions(
    model_seed,
    source_model,
    encoder,
    train,
    test,
    initial_state,
    steps,
    batch_size,
    learning_rate,
    device,
):
    args = SimpleNamespace(
        steps=steps,
        batch_size=batch_size,
        lr=learning_rate,
    )
    raw = train_raw(model_seed, train, test, args, device, initial_state)
    direct = reader.train_direct_int4(
        model_seed, source_model, encoder, train, test, args, device, initial_state
    )
    soft = reader.train_soft_reader(
        model_seed, source_model, encoder, train, test, args, device, initial_state
    )
    decoded_train = reader.decode_int4(source_model, encoder, train)
    decoded_test = reader.decode_int4(source_model, encoder, test)
    decoded = train_raw(
        model_seed, decoded_train, decoded_test, args, device, initial_state
    )

    return {
        "steps": steps,
        "direct_int4_latent_thinking": direct,
        "soft_reader_int4_thinking": soft,
        "argmax_decoded_int4_thinking": decoded,
        "raw_token_baseline": raw,
        "paired_test_deltas_pp": {
            "direct_minus_raw": 100 * (
                direct["test"]["accuracy"] - raw["accuracy"]
            ),
            "soft_reader_minus_raw": 100 * (
                soft["test"]["accuracy"] - raw["accuracy"]
            ),
            "argmax_decoded_minus_raw": 100 * (
                decoded["accuracy"] - raw["accuracy"]
            ),
        },
    }


def write_report(report, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--budgets", nargs="+", type=int, default=[800, 1600])
    parser.add_argument("--autoencoder-steps", type=int, default=AUTOENCODER_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output",
        default="experiments/latent_chunk/fixed_data_offset4_rank4_budget_robustness.json",
    )
    cli = parser.parse_args()
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    train, test, held_out_pairs = reader.split_offset_four(cli.data_seed, device)
    report = {
        "task": "two_phrase_key_value_relation_query_with_counterfactual_pairs",
        "condition": "fixed_offset4_data_seed_independent_model_seeds_rank4_int4",
        "method": (
            "Sample the offset-4 train/test split once from data_seed. For each independent "
            "model_seed, fit one phrase autoencoder for autoencoder_steps, then train raw-token, "
            "direct int4, soft-reader, and argmax-decoded task conditions at each task budget. "
            "All task arms share the same initial PBSLM state and batch-index sequence per "
            "model_seed and budget."
        ),
        "split": {
            "data_seed": cli.data_seed,
            "model_seeds": cli.model_seeds,
            "held_out_pairs": held_out_pairs,
            "train_examples": train["target"].numel(),
            "test_examples": test["target"].numel(),
            "source": (
                "offset-4 rows removed from the sampled training pool before phrase "
                "autoencoder or task-model training"
            ),
        },
        "latent_code": {
            "rank_per_phrase": RANK,
            "width": 32,
            "bits_per_component": 4,
            "payload_bytes_bit_packed_estimate": 2 * RANK * 32 * 4 // 8,
            "runtime_quantization": "dequantized FP32 levels from round(z*7).clamp(-7,7)/7",
            "packed_kernel_or_runtime_memory_savings_measured": False,
        },
        "training": {
            "autoencoder_steps": cli.autoencoder_steps,
            "task_budgets": cli.budgets,
            "batch_size": cli.batch_size,
            "learning_rate": cli.lr,
            "optimizer": "AdamW",
        },
        "variance_design": (
            "The data split is held fixed while model initialization seeds vary. Results "
            "therefore estimate model-seed and optimizer variation conditional on one "
            "sampled synthetic dataset, not data-sampling variance."
        ),
        "limitations": (
            "Synthetic task with explicit x/y segmentation and frozen random source embeddings. "
            "The soft reader uses a shared decoder and expands codes to 16 continuous vectors. "
            "Only one data seed is used, so results do not estimate variability across sampled "
            "datasets or natural-language performance."
        ),
        "config": vars(cli),
        "progress": {
            "completed_seed_budget_runs": 0,
            "total_seed_budget_runs": len(cli.model_seeds) * len(cli.budgets),
            "complete": False,
        },
        "torch": torch.__version__,
        "provenance": capture([
            "evaluation/latent_chunk_fixed_data_budget.py",
            "evaluation/latent_chunk_soft_reader_confirmatory.py",
            "evaluation/latent_chunk_reconstruction.py",
            "evaluation/latent_chunk_ab_thinking.py",
            "evaluation/latent_chunk_relations.py",
            "evaluation/provenance.py",
            "models/model.py",
            "models/model_v2.py",
        ]),
        "model_runs": [],
    }
    output_path = ROOT / cli.output

    for model_seed in cli.model_seeds:
        initial_model = new_model(model_seed, device)
        initial_state = copy.deepcopy(initial_model.state_dict())
        del initial_model

        autoencoder_args = SimpleNamespace(
            steps=cli.autoencoder_steps,
            batch_size=cli.batch_size,
            lr=cli.lr,
        )
        source_model, encoder, reconstruction = train_autoencoder(
            model_seed,
            RANK,
            train,
            test,
            autoencoder_args,
            device,
            initial_state,
        )
        decoded_test = reader.decode_int4(source_model, encoder, test)
        int4_reconstruction = reader.token_reconstruction_metrics(
            decoded_test["ids"], test["ids"]
        )
        model_record = {
            "data_seed": cli.data_seed,
            "model_seed": model_seed,
            "train_examples": train["target"].numel(),
            "test_examples": test["target"].numel(),
            "latent_payload_bytes_int4_estimate": 2 * RANK * 32 * 4 // 8,
            "autoencoder_parameters": sum(
                parameter.numel() for parameter in encoder.parameters()
            ),
            "shared_encoder_parameters": sum(
                parameter.numel() for parameter in encoder.encoder.parameters()
            ),
            "shared_decoder_parameters": sum(
                parameter.numel() for parameter in encoder.decoder.parameters()
            ),
            "full_precision_reconstruction": reconstruction,
            "int4_quantized_test_reconstruction": int4_reconstruction,
            "task_budgets": [],
        }
        report["model_runs"].append(model_record)
        write_report(report, output_path)

        for steps in cli.budgets:
            budget_result = run_task_conditions(
                model_seed,
                source_model,
                encoder,
                train,
                test,
                initial_state,
                steps,
                cli.batch_size,
                cli.lr,
                device,
            )
            model_record["task_budgets"].append(budget_result)
            report["progress"]["completed_seed_budget_runs"] += 1
            write_report(report, output_path)
            print(
                json.dumps(
                    {
                        "model_seed": model_seed,
                        "steps": steps,
                        "paired_test_deltas_pp": budget_result[
                            "paired_test_deltas_pp"
                        ],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        del source_model, encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report["progress"]["complete"] = True
    write_report(report, output_path)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
