"""Test whether compact-reader outputs remain usable as packed int4 memory."""
import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from evaluation.latent_chunk_ab_thinking import class_logits, new_model, train_raw
from evaluation.latent_chunk_compact_reader import (
    CompactPairReader,
    compact_embeddings,
    make_offsets_sets,
)
from evaluation.latent_chunk_holdout import make_held_out_data
from evaluation.latent_chunk_reconstruction import LatentRoleTags, train_autoencoder
from evaluation.latent_chunk_soft_reader_confirmatory import RANK
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]
CODE_WIDTH = 2 * RANK * 32


def fake_int4_ste(embeddings):
    quantized = (embeddings * 7).round().clamp(-7, 7) / 7
    return embeddings + (quantized - embeddings).detach()


def pack_int4(embeddings):
    values = (embeddings * 7).round().clamp(-7, 7).to(torch.int16)
    nibbles = torch.bitwise_and(values.flatten(1), 0xF)
    low = nibbles[:, 0::2].to(torch.uint8)
    high = torch.bitwise_left_shift(nibbles[:, 1::2].to(torch.uint8), 4)
    return torch.bitwise_or(low, high)


def unpack_int4(packed, shape):
    low = torch.bitwise_and(packed, 0xF).to(torch.int16)
    high = torch.bitwise_and(torch.bitwise_right_shift(packed, 4), 0xF).to(torch.int16)
    values = torch.stack((low, high), dim=-1).flatten(1)
    signed = torch.where(values >= 8, values - 16, values)
    return (signed.reshape(shape).float() / 7)


def evaluate_compact(model, source_model, encoder, reader, tags, data, quantized):
    model.eval()
    source_model.eval()
    encoder.eval()
    reader.eval()
    tags.eval()
    predictions = []
    code_bytes = 0
    with torch.no_grad():
        for start in range(0, data["target"].numel(), 256):
            sl = slice(start, min(start + 256, data["target"].numel()))
            embeddings = compact_embeddings(
                model, source_model, encoder, reader, tags, data["ids"][sl]
            )
            if quantized:
                packed = pack_int4(embeddings)
                code_bytes += packed.numel()
                embeddings = unpack_int4(packed, embeddings.shape)
            output = model(None, 1, inputs_embeds=embeddings, sample_noise=False)
            predictions.extend(class_logits(model, output).argmax(-1).tolist())

    predictions = torch.tensor(predictions, device=data["target"].device)
    pair_exact = (
        predictions[0::2].eq(data["target"][0::2])
        & predictions[1::2].eq(data["target"][1::2])
    ).float().mean()
    result = {
        "accuracy": float((predictions == data["target"]).float().mean()),
        "counterfactual_pair_exact": float(pair_exact),
        "examples": data["target"].numel(),
    }
    if quantized:
        result["packed_code_bytes_total"] = code_bytes
        result["packed_code_bytes_per_example"] = code_bytes / data["target"].numel()
    return result


def train_compact_float(seed, source_model, encoder, train, test, args, device, initial_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_state)
    tags = LatentRoleTags(RANK, model.d).to(device)
    reader = CompactPairReader(model.d).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable.extend(reader.parameters())
    trainable.extend(tags.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    torch.manual_seed(seed * 1009 + 7)
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 19)
    model.train()
    reader.train()
    tags.train()

    for _ in range(args.steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        ids = train["ids"][indices]
        embeddings = compact_embeddings(
            model, source_model, encoder, reader, tags, ids
        )
        output = model(None, 1, inputs_embeds=embeddings, sample_noise=False)
        objective = F.cross_entropy(class_logits(model, output), train["target"][indices])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    return {
        "test": evaluate_compact(
            model, source_model, encoder, reader, tags, test, quantized=False
        ),
        "test_posthoc_packed_int4": evaluate_compact(
            model, source_model, encoder, reader, tags, test, quantized=True
        ),
        "train": evaluate_compact(
            model, source_model, encoder, reader, tags, train, quantized=False
        ),
        "input_vectors": 2 * RANK,
        "additional_reader_parameters": sum(p.numel() for p in reader.parameters()),
        "role_tag_parameters": sum(p.numel() for p in tags.parameters()),
        "trainable_model_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }


def train_compact_int4(seed, source_model, encoder, train, test, args, device, initial_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_state)
    tags = LatentRoleTags(RANK, model.d).to(device)
    reader = CompactPairReader(model.d).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable.extend(reader.parameters())
    trainable.extend(tags.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    torch.manual_seed(seed * 1009 + 7)
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 19)
    model.train()
    reader.train()
    tags.train()

    for _ in range(args.steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        ids = train["ids"][indices]
        embeddings = compact_embeddings(
            model, source_model, encoder, reader, tags, ids
        )
        output = model(
            None, 1, inputs_embeds=fake_int4_ste(embeddings), sample_noise=False
        )
        objective = F.cross_entropy(class_logits(model, output), train["target"][indices])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    return {
        "test": evaluate_compact(
            model, source_model, encoder, reader, tags, test, quantized=True
        ),
        "train": evaluate_compact(
            model, source_model, encoder, reader, tags, train, quantized=True
        ),
        "input_vectors": 2 * RANK,
        "packed_int4_bytes_per_example": CODE_WIDTH // 2,
        "additional_reader_parameters": sum(p.numel() for p in reader.parameters()),
        "role_tag_parameters": sum(p.numel() for p in tags.parameters()),
        "trainable_model_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }


def mean_sd(values):
    if len(values) == 1:
        return values[0], 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def summarize(report):
    lines = [
        "# Packed Int4 Compact Reader",
        "",
        f"- Status: {str(report['progress']['complete']).lower()} ({len(report['runs'])} completed runs)",
        f"- Model seeds: {report['config']['seeds']}",
        f"- Task steps per arm: {report['config']['steps']}",
        "- Results are mean ± sample SD across model seeds on each fixed balanced split.",
        "",
        "| Held-out offsets | Input | Test accuracy | Counterfactual pair exact |",
        "|---|---|---:|---:|",
    ]
    for offsets in sorted({tuple(run["held_out_offsets"]) for run in report["runs"]}):
        runs = [run for run in report["runs"] if tuple(run["held_out_offsets"]) == offsets]
        for arm, label in (
            ("raw_token_baseline", "Raw tokens"),
            ("compact_float_reader", "Compact FP32 vectors"),
            ("compact_float_posthoc_int4", "FP32-trained vectors, packed int4 at test"),
            ("compact_int4_reader", "Quantization-aware trained vectors"),
        ):
            if arm == "raw_token_baseline":
                metrics = [run[arm] for run in runs]
            elif arm == "compact_float_posthoc_int4":
                metrics = [run["compact_float_reader"]["test_posthoc_packed_int4"] for run in runs]
            else:
                metrics = [run[arm]["test"] for run in runs]
            accuracy_mean, accuracy_sd = mean_sd([item["accuracy"] for item in metrics])
            pair_mean, pair_sd = mean_sd([item["counterfactual_pair_exact"] for item in metrics])
            lines.append(
                f"| {list(offsets)} | {label} | {100 * accuracy_mean:.2f} ± {100 * accuracy_sd:.2f}% | "
                f"{100 * pair_mean:.2f} ± {100 * pair_sd:.2f}% |"
            )
    lines.extend([
        "",
        "The post-hoc packed row uses the same checkpoint as the FP32 row, isolating int4 quantization at inference. The quantization-aware row trains with fake-int4 quantization and a straight-through gradient. Evaluation packs signed 4-bit values into bytes and unpacks those bytes before the trunk. Each example stores eight 32-dimensional vectors in 128 bytes, excluding shared encoder, merger, role tags, and trunk parameters.",
        "",
        "The compact representation is produced from the decoded phrase by a learned pair merger and role tags. The experiment tests quantized task-specific memory vectors in this synthetic task. The source phrase decoder is still used when writing the representation. Ideal fixed-width packing of the 16 source IDs requires 8 bytes, so this is not a storage win over source text. It does not measure a production runtime cache or latency.",
        "",
    ])
    return "\n".join(lines)


def write_json_atomic(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--held-out-offsets", nargs="+", type=int, default=[4, 3, 5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--pretrain-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        default="experiments/latent_chunk/compact_reader_int4_data0_model0-4_steps800.json",
    )
    cli = parser.parse_args()
    if min(cli.steps, cli.pretrain_steps, cli.batch_size) <= 0:
        parser.error("steps, pretrain steps, and batch size must be positive")
    if not cli.seeds or len(set(cli.seeds)) != len(cli.seeds):
        parser.error("seeds must be nonempty and unique")
    if not cli.held_out_offsets or any(offset not in range(1, 8) for offset in cli.held_out_offsets):
        parser.error("held-out offsets must be a nonempty list from 1 to 7")
    if len(set(cli.held_out_offsets)) != len(cli.held_out_offsets):
        parser.error("held-out offsets must not contain duplicates")

    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    args = SimpleNamespace(steps=cli.steps, batch_size=cli.batch_size, lr=cli.lr)
    output_path = ROOT / cli.output
    offsets_sets = [tuple(offsets) for offsets in make_offsets_sets(cli.held_out_offsets)]
    config = vars(cli).copy()
    config.pop("resume")
    source_paths = [
        "evaluation/latent_chunk_compact_int4.py",
        "evaluation/latent_chunk_compact_reader.py",
        "evaluation/latent_chunk_soft_reader_confirmatory.py",
        "evaluation/latent_chunk_reconstruction.py",
        "evaluation/latent_chunk_ab_thinking.py",
        "evaluation/latent_chunk_holdout.py",
        "evaluation/latent_chunk_relations.py",
        "evaluation/provenance.py",
        "models/model_v2.py",
    ]
    provenance = capture(source_paths)
    if output_path.exists():
        if not cli.resume:
            parser.error(f"output already exists; pass --resume to continue: {output_path}")
        report = json.loads(output_path.read_text(encoding="utf-8"))
        if report["config"] != config:
            parser.error("resume configuration does not match the existing result")
        if report["provenance"]["source_sha256_at_capture"] != provenance["source_sha256_at_capture"]:
            parser.error("resume source hashes do not match the existing result")
        if any(
            not {"raw_token_baseline", "compact_float_reader", "compact_int4_reader"}.issubset(run)
            for run in report["runs"]
        ):
            parser.error("resume result is missing a required condition")
    else:
        report = {
            "task": "quantized_compact_latent_reader",
            "method": "Compare a matched raw-token trunk, a task-trained FP32 compact reader with FP32 and post-hoc packed evaluations from the same checkpoint, and a quantization-aware compact reader evaluated through actual packed int4 bytes.",
            "config": config,
            "storage": {
                "packed_int4_bytes_per_example": CODE_WIDTH // 2,
                "ideal_packed_synthetic_token_ids_bytes": 8,
                "shared_reader_encoder_trunk_excluded": True,
            },
            "limitations": "Synthetic held-out relation task. Phrase encoding and decoding generate task-specific compact vectors at write time. FP32 trunk inputs are dequantized from packed int4 bytes during evaluation. No production runtime cache, end-to-end memory or latency measurement is performed.",
            "provenance": provenance,
            "progress": {"complete": False},
            "runs": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_path, report)

    completed = {(tuple(run["held_out_offsets"]), run["seed"]) for run in report["runs"]}
    for offsets in offsets_sets:
        train, test, held_out_pairs = make_held_out_data(
            cli.data_seed, device, held_out_offsets=offsets
        )
        for seed in cli.seeds:
            run_key = (offsets, seed)
            if run_key in completed:
                continue
            print(f"offsets {offsets}, model seed {seed}: preparing matched int4 conditions", flush=True)
            initial_model = new_model(seed, device)
            initial_state = copy.deepcopy(initial_model.state_dict())
            del initial_model
            raw = train_raw(seed, train, test, args, device, initial_state)
            source_model, encoder, reconstruction = train_autoencoder(
                seed, RANK, train, test,
                SimpleNamespace(
                    steps=cli.pretrain_steps, batch_size=cli.batch_size, lr=cli.lr
                ),
                device, initial_state,
            )
            compact_float = train_compact_float(
                seed, source_model, encoder, train, test, args, device, initial_state
            )
            compact_int4 = train_compact_int4(
                seed, source_model, encoder, train, test, args, device, initial_state
            )
            report["runs"].append({
                "held_out_offsets": list(offsets),
                "held_out_pairs": held_out_pairs,
                "seed": seed,
                "train_examples": train["target"].numel(),
                "test_examples": test["target"].numel(),
                "reconstruction_pretraining": reconstruction,
                "raw_token_baseline": raw,
                "compact_float_reader": compact_float,
                "compact_int4_reader": compact_int4,
            })
            completed.add(run_key)
            report["progress"]["complete"] = len(completed) == len(offsets_sets) * len(cli.seeds)
            write_json_atomic(output_path, report)
            print(json.dumps(report["runs"][-1], ensure_ascii=False), flush=True)
            del source_model, encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report["progress"]["complete"] = len(completed) == len(offsets_sets) * len(cli.seeds)
    write_json_atomic(output_path, report)
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(summarize(report), encoding="utf-8")
    print(f"wrote {output_path}", flush=True)
    print(f"wrote {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
