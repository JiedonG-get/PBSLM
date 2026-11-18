"""Test task-aware joint tuning of int4-quantized phrase latents.

The encoder and decoder are first trained for phrase reconstruction. They are
then jointly tuned with the PBSLM trunk under task loss plus full-phrase
reconstruction loss. The quantized codes use a straight-through estimator.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluation.latent_chunk_ab_thinking import class_logits, new_model
from evaluation.latent_chunk_reconstruction import (
    LatentRoleTags,
    train_autoencoder,
)
from evaluation.latent_chunk_relations import CLASSES, PHRASE_LENGTH
from evaluation.latent_chunk_soft_reader_confirmatory import (
    BITS,
    RANK,
    quantize_int4,
    split_offset_four,
)
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCTION_WEIGHT = 1.0


def quantize_int4_ste(latent):
    quantized = quantize_int4(latent)
    return latent + (quantized - latent).detach()


def evaluate_task(model, source_model, encoder, tags, data, batch_size=256):
    model.eval()
    source_model.eval()
    encoder.eval()
    tags.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, data["target"].numel(), batch_size):
            sl = slice(start, min(start + batch_size, data["target"].numel()))
            a, b = encoder.encode(source_model, data["ids"][sl])
            embeds = tags.apply(quantize_int4(a), quantize_int4(b))
            output = model(None, 1, inputs_embeds=embeds, sample_noise=False)
            predictions.extend(class_logits(model, output).argmax(-1).tolist())

    predictions = torch.tensor(predictions, device=data["target"].device)
    pair_exact = (
        predictions[0::2].eq(data["target"][0::2])
        & predictions[1::2].eq(data["target"][1::2])
    ).float().mean()
    ordered_pairs = data["ids"][:, PHRASE_LENGTH + 2:PHRASE_LENGTH + 5:2]
    per_pair_accuracy = {}
    for pair in torch.unique(ordered_pairs, dim=0):
        selected = ordered_pairs.eq(pair).all(-1)
        pair_name = "->".join(str(value) for value in pair.tolist())
        per_pair_accuracy[pair_name] = float(
            (predictions[selected] == data["target"][selected]).float().mean()
        )
    return {
        "accuracy": float((predictions == data["target"]).float().mean()),
        "counterfactual_pair_exact": float(pair_exact),
        "ordered_value_pair_accuracy": per_pair_accuracy,
        "examples": int(data["target"].numel()),
    }


@torch.no_grad()
def evaluate_raw_task(model, data, batch_size=256):
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
    ordered_pairs = data["ids"][:, PHRASE_LENGTH + 2:PHRASE_LENGTH + 5:2]
    per_pair_accuracy = {}
    for pair in torch.unique(ordered_pairs, dim=0):
        selected = ordered_pairs.eq(pair).all(-1)
        pair_name = "->".join(str(value) for value in pair.tolist())
        per_pair_accuracy[pair_name] = float(
            (predictions[selected] == data["target"][selected]).float().mean()
        )
    return {
        "accuracy": float((predictions == data["target"]).float().mean()),
        "counterfactual_pair_exact": float(pair_exact),
        "ordered_value_pair_accuracy": per_pair_accuracy,
        "examples": int(data["target"].numel()),
    }


def train_raw_with_pair_metrics(seed, train, test, args, device, initial_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_state)
    torch.manual_seed(seed * 1009 + 7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 19)
    model.train()
    started = time.perf_counter()

    for _ in range(args.steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        batch = {key: value[indices] for key, value in train.items()}
        output = model(batch["ids"], 1, sample_noise=False)
        objective = F.cross_entropy(class_logits(model, output), batch["target"])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    result = evaluate_raw_task(model, test)
    result["train_evaluation"] = evaluate_raw_task(model, train)
    result["train_seconds"] = time.perf_counter() - started
    return result


@torch.no_grad()
def evaluate_int4_reconstruction(source_model, encoder, data, batch_size=256):
    source_model.eval()
    encoder.eval()
    token_correct = torch.zeros(2 * PHRASE_LENGTH, device=data["ids"].device)
    x_exact = 0
    y_exact = 0
    joint_exact = 0
    total = int(data["target"].numel())

    for start in range(0, total, batch_size):
        sl = slice(start, min(start + batch_size, total))
        ids = data["ids"][sl]
        a, b = encoder.encode(source_model, ids)
        x_prediction = encoder.decode(quantize_int4(a)).argmax(-1)
        y_prediction = encoder.decode(quantize_int4(b)).argmax(-1)
        predicted = torch.cat((x_prediction, y_prediction), dim=-1)
        token_correct += (predicted == ids).sum(0)
        x_match = (x_prediction == ids[:, :PHRASE_LENGTH]).all(-1)
        y_match = (y_prediction == ids[:, PHRASE_LENGTH:]).all(-1)
        x_exact += int(x_match.sum())
        y_exact += int(y_match.sum())
        joint_exact += int((x_match & y_match).sum())

    accuracies = (token_correct / total).tolist()
    return {
        "all_16_token_accuracy": sum(accuracies) / (2 * PHRASE_LENGTH),
        "x_token_accuracy": sum(accuracies[:PHRASE_LENGTH]) / PHRASE_LENGTH,
        "y_token_accuracy": sum(accuracies[PHRASE_LENGTH:]) / PHRASE_LENGTH,
        "query_key_accuracy": accuracies[1],
        "y_edge_symbol_accuracy": sum(accuracies[PHRASE_LENGTH + 1:PHRASE_LENGTH + 5]) / 4,
        "x_phrase_exact": x_exact / total,
        "y_phrase_exact": y_exact / total,
        "both_phrases_exact": joint_exact / total,
        "examples": total,
    }


def train_frozen_int4(seed, source_model, encoder, train, test, args, device, initial_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_state)
    tags = LatentRoleTags(RANK, model.d).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable.extend(tags.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    torch.manual_seed(seed * 1009 + 7)
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 19)
    model.train()
    encoder.eval()
    tags.train()
    started = time.perf_counter()

    for _ in range(args.steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        ids = train["ids"][indices]
        with torch.no_grad():
            a, b = encoder.encode(source_model, ids)
            embeds = tags.apply(quantize_int4(a), quantize_int4(b))
        output = model(None, 1, inputs_embeds=embeds, sample_noise=False)
        objective = F.cross_entropy(class_logits(model, output), train["target"][indices])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    return {
        "train": evaluate_task(model, source_model, encoder, tags, train),
        "test": evaluate_task(model, source_model, encoder, tags, test),
        "final_batch_task_loss": float(objective.detach()),
        "train_seconds": time.perf_counter() - started,
        "trainable_trunk_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "trainable_role_tag_parameters": sum(parameter.numel() for parameter in tags.parameters()),
    }


def train_joint(seed, source_model, encoder, train, test, args, device, initial_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_state)
    tags = LatentRoleTags(RANK, model.d).to(device)
    model.train()
    encoder.train()
    tags.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable.extend(encoder.parameters())
    trainable.extend(tags.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    torch.manual_seed(seed * 1009 + 7)
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 19)
    started = time.perf_counter()

    for step in range(args.joint_steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        ids = train["ids"][indices]
        a, b = encoder.encode(source_model, ids)
        a_quantized = quantize_int4_ste(a)
        b_quantized = quantize_int4_ste(b)

        x_logits = encoder.decode(a_quantized)
        y_logits = encoder.decode(b_quantized)
        reconstruction_loss = (
            F.cross_entropy(x_logits.flatten(0, 1), ids[:, :PHRASE_LENGTH].flatten())
            + F.cross_entropy(y_logits.flatten(0, 1), ids[:, PHRASE_LENGTH:].flatten())
        ) / 2

        embeds = tags.apply(a_quantized, b_quantized)
        output = model(None, 1, inputs_embeds=embeds, sample_noise=False)
        task_loss = F.cross_entropy(class_logits(model, output), train["target"][indices])
        objective = task_loss + RECONSTRUCTION_WEIGHT * reconstruction_loss
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    return {
        "train": evaluate_task(model, source_model, encoder, tags, train),
        "test": evaluate_task(model, source_model, encoder, tags, test),
        "int4_train_reconstruction": evaluate_int4_reconstruction(source_model, encoder, train),
        "int4_test_reconstruction": evaluate_int4_reconstruction(source_model, encoder, test),
        "final_batch_task_loss": float(task_loss.detach()),
        "final_batch_reconstruction_loss": float(reconstruction_loss.detach()),
        "train_seconds": time.perf_counter() - started,
        "trainable_trunk_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "trainable_autoencoder_parameters": sum(
            parameter.numel() for parameter in encoder.parameters()
        ),
        "trainable_role_tag_parameters": sum(parameter.numel() for parameter in tags.parameters()),
    }


def atomic_write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".tmp.", delete=False
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def artifact_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_model_seed(model_seed, train, test, args, device):
    initial_model = new_model(model_seed, device)
    initial_state = copy.deepcopy(initial_model.state_dict())
    del initial_model

    baseline_args = argparse.Namespace(
        steps=args.baseline_steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"model seed {model_seed}: starting raw-token baseline", flush=True)
    raw_baseline = train_raw_with_pair_metrics(
        model_seed, train, test, baseline_args, device, initial_state
    )
    print(
        f"model seed {model_seed}: raw baseline finished in {raw_baseline['train_seconds']:.1f}s",
        flush=True,
    )

    pretrain_args = argparse.Namespace(
        steps=args.pretrain_steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"model seed {model_seed}: starting phrase autoencoder pretraining", flush=True)
    source_model, encoder, pretraining = train_autoencoder(
        model_seed, RANK, train, test, pretrain_args, device, initial_state
    )
    pretraining_int4_reconstruction = {
        "train": evaluate_int4_reconstruction(source_model, encoder, train),
        "test": evaluate_int4_reconstruction(source_model, encoder, test),
    }
    print(
        f"model seed {model_seed}: autoencoder finished in {pretraining['train_seconds']:.1f}s",
        flush=True,
    )

    frozen_args = argparse.Namespace(
        steps=args.frozen_steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"model seed {model_seed}: starting frozen int4-latent control", flush=True)
    frozen_result = train_frozen_int4(
        model_seed, source_model, encoder, train, test, frozen_args, device, initial_state
    )
    print(
        f"model seed {model_seed}: frozen control finished in {frozen_result['train_seconds']:.1f}s",
        flush=True,
    )

    joint_args = argparse.Namespace(
        joint_steps=args.joint_steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"model seed {model_seed}: starting joint reconstruction training", flush=True)
    joint_result = train_joint(
        model_seed, source_model, encoder, train, test, joint_args, device, initial_state
    )
    print(
        f"model seed {model_seed}: joint training finished in {joint_result['train_seconds']:.1f}s",
        flush=True,
    )

    return {
        "model_seed": model_seed,
        "train_examples": int(train["target"].numel()),
        "test_examples": int(test["target"].numel()),
        "autoencoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "shared_encoder_parameters": sum(parameter.numel() for parameter in encoder.encoder.parameters()),
        "shared_decoder_parameters": sum(parameter.numel() for parameter in encoder.decoder.parameters()),
        "pretraining": pretraining,
        "pretraining_int4_reconstruction": pretraining_int4_reconstruction,
        "frozen_int4_latent": frozen_result,
        "joint_int4_latent": joint_result,
        "raw_token_baseline": raw_baseline,
    }


def write_markdown(report, path):
    status = "complete" if report["progress"]["complete"] else "in progress"
    split = report["split"]
    config = report["config"]
    lines = [
        "# Task-Aware Joint Reconstruction Experiment",
        "",
        f"- Status: `{status}` ({len(report['runs'])}/{len(config['model_seeds'])} model seeds completed)",
        f"- Data seed: `{config['data_seed']}`",
        f"- Model seeds requested: `{config['model_seeds']}`",
        f"- Split: offset-4 ordered value pairs held out before training ({split['train_examples']} train, {split['test_examples']} test rows)",
        f"- Rank: `{RANK}` per phrase, with `{2 * RANK}` input vectors total",
        f"- Steps: `{config['pretrain_steps']}` reconstruction pretraining, `{config['frozen_steps']}` frozen-latent trunk training, `{config['joint_steps']}` joint tuning, `{config['baseline_steps']}` raw baseline",
        f"- Objective: task cross-entropy + `{RECONSTRUCTION_WEIGHT}` x full-phrase reconstruction cross-entropy",
        "- Quantization: signed int4-equivalent fake quantization, with FP32 tensors and straight-through gradients",
        "",
        "## Task Results",
        "",
        "| Model seed | Raw train | Raw test | Frozen train | Frozen test | Joint train | Joint test | Raw pair-exact | Frozen pair-exact | Joint pair-exact |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for record in report["runs"]:
        raw = record["raw_token_baseline"]
        frozen = record["frozen_int4_latent"]
        joint = record["joint_int4_latent"]
        lines.append(
            f"| {record['model_seed']} | {raw['train_evaluation']['accuracy']:.4f} | {raw['accuracy']:.4f} | "
            f"{frozen['train']['accuracy']:.4f} | {frozen['test']['accuracy']:.4f} | "
            f"{joint['train']['accuracy']:.4f} | {joint['test']['accuracy']:.4f} | "
            f"{raw['counterfactual_pair_exact']:.4f} | {frozen['test']['counterfactual_pair_exact']:.4f} | "
            f"{joint['test']['counterfactual_pair_exact']:.4f} |"
        )

    lines.extend([
        "",
        "## Int4 Reconstruction",
        "",
        "| Model seed | Before task tuning, token acc. | Before, both phrases exact | After joint tuning, token acc. | After, both phrases exact |",
        "|---:|---:|---:|---:|---:|",
    ])
    for record in report["runs"]:
        before = record["pretraining_int4_reconstruction"]["test"]
        after = record["joint_int4_latent"]["int4_test_reconstruction"]
        lines.append(
            f"| {record['model_seed']} | {before['all_16_token_accuracy']:.4f} | "
            f"{before['both_phrases_exact']:.4f} | {after['all_16_token_accuracy']:.4f} | "
            f"{after['both_phrases_exact']:.4f} |"
        )

    for record in report["runs"]:
        seed = record["model_seed"]
        lines.extend([
            "",
            f"## Held-Out Pair Accuracy, Model Seed {seed}",
            "",
            "| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |",
            "|---|---:|---:|---:|",
        ])
        raw_pairs = record["raw_token_baseline"]["ordered_value_pair_accuracy"]
        frozen_pairs = record["frozen_int4_latent"]["test"]["ordered_value_pair_accuracy"]
        joint_pairs = record["joint_int4_latent"]["test"]["ordered_value_pair_accuracy"]
        for pair in ("->".join(map(str, values)) for values in split["held_out_ordered_value_pairs"]):
            lines.append(
                f"| {pair} | {raw_pairs[pair]:.4f} | {frozen_pairs[pair]:.4f} | {joint_pairs[pair]:.4f} |"
            )

    lines.extend([
        "",
        "The experiment uses a synthetic two-phrase relation task and frozen randomly initialized source token embeddings. The int4 values are dequantized to FP32 for training and inference. The 128-byte latent payload is larger than the estimated 8-byte 4-bit-packed payload for the original 16 token IDs, though smaller than their 2048-byte FP32 embedding input. Shared encoder/decoder storage is excluded, and no packed-runtime memory or latency savings are measured.",
        "",
        "Each completed seed has an atomic JSON checkpoint. The aggregate JSON records checkpoint paths and SHA-256 digests, plus source hashes.",
        "",
    ])
    atomic_write_text(path, "\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--pretrain-steps", type=int, default=800)
    parser.add_argument("--frozen-steps", type=int, default=800)
    parser.add_argument("--joint-steps", type=int, default=800)
    parser.add_argument("--baseline-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="experiments/latent_chunk/task_aware_joint_recon_int4_offset4_data0_model0-4.json",
    )
    args = parser.parse_args()
    if len(set(args.model_seeds)) != len(args.model_seeds):
        parser.error("--model-seeds must not contain duplicates")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    train, test, held_out_pairs = split_offset_four(args.data_seed, device)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    checkpoint_dir = output_path.parent / f"{output_path.stem}_checkpoints"

    report = {
        "task": "two_phrase_relation_query_with_offset4_counterfactual_holdout",
        "condition": "frozen_vs_joint_int4_a_b_latent_task_tuning_with_phrase_reconstruction",
        "method": "Pretrain a shared rank-4 phrase autoencoder, evaluate int4 reconstruction, then compare a frozen-latent task-only trunk against a matched jointly tuned encoder, decoder, role tags, and PBSLM trunk using task CE plus full-phrase reconstruction CE.",
        "limitations": "Synthetic relation task; frozen random source token embeddings; one predeclared offset-4 split with fixed data seed; fake quantization executes with FP32 tensors and does not measure packed runtime memory or latency.",
        "config": {**vars(args), "reconstruction_weight": RECONSTRUCTION_WEIGHT},
        "torch": torch.__version__,
        "provenance": capture([
            "evaluation/latent_chunk_task_aware_joint_reconstruction.py",
            "evaluation/latent_chunk_soft_reader_confirmatory.py",
            "evaluation/latent_chunk_reconstruction.py",
            "evaluation/latent_chunk_ab_thinking.py",
            "evaluation/latent_chunk_relations.py",
            "evaluation/provenance.py",
            "models/model.py",
            "models/model_v2.py",
        ]),
        "code_storage": {
            "bits_per_component": BITS,
            "rank_per_phrase": RANK,
            "phrase_count": 2,
            "components": 2 * RANK * 32,
            "int4_payload_bytes_estimate": 2 * RANK * 32 * BITS // 8,
            "dequantized_fp32_input_bytes": 2 * RANK * 32 * 4,
            "raw_input_token_count": 2 * PHRASE_LENGTH,
            "input_token_bit_width": math.ceil(math.log2(CLASSES + 2)),
            "raw_token_payload_bytes_4bit_packed_estimate": (
                2 * PHRASE_LENGTH * math.ceil(math.log2(CLASSES + 2)) // 8
            ),
            "raw_fp32_embedding_input_bytes": 2 * PHRASE_LENGTH * 32 * 4,
            "latent_to_bitpacked_token_payload_ratio": (
                (2 * RANK * 32 * BITS)
                / (2 * PHRASE_LENGTH * math.ceil(math.log2(CLASSES + 2)))
            ),
            "shared_autoencoder_and_decoder_excluded": True,
        },
        "split": {
            "data_seed": args.data_seed,
            "held_out_ordered_value_pairs": held_out_pairs,
            "train_examples": int(train["target"].numel()),
            "test_examples": int(test["target"].numel()),
            "standard_test_split_used": False,
        },
        "progress": {
            "complete": False,
            "requested_model_seeds": list(args.model_seeds),
            "completed_model_seeds": [],
            "checkpoint_files": {},
        },
        "runs": [],
    }

    atomic_write_json(output_path, report)
    for model_seed in args.model_seeds:
        run = train_model_seed(model_seed, train, test, args, device)
        checkpoint_path = checkpoint_dir / f"model_seed_{model_seed}.json"
        checkpoint = {
            "task": report["task"],
            "data_seed": args.data_seed,
            "model_seed": model_seed,
            "config": report["config"],
            "split": report["split"],
            "code_storage": report["code_storage"],
            "provenance": report["provenance"],
            "progress": {"complete": True, "checkpoint_model_seed": model_seed},
            "run": run,
        }
        atomic_write_json(checkpoint_path, checkpoint)
        try:
            checkpoint_relative_path = str(checkpoint_path.relative_to(ROOT))
        except ValueError:
            checkpoint_relative_path = str(checkpoint_path)
        checkpoint_hash = artifact_sha256(checkpoint_path)
        report["runs"].append(run)
        report["progress"]["completed_model_seeds"].append(model_seed)
        report["progress"]["checkpoint_files"][str(model_seed)] = {
            "path": checkpoint_relative_path,
            "sha256": checkpoint_hash,
        }
        report["progress"]["complete"] = (
            report["progress"]["completed_model_seeds"]
            == report["progress"]["requested_model_seeds"]
        )
        atomic_write_json(output_path, report)
        write_markdown(report, output_path.with_suffix(".md"))
        print(
            f"completed model seed {model_seed}; checkpoint sha256={checkpoint_hash}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"wrote aggregate report {output_path}", flush=True)


if __name__ == "__main__":
    main()
