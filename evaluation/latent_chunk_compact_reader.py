"""Compare an 8-vector learned reader with direct codes and 16-vector inputs."""
import argparse
import copy
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.latent_chunk_ab_thinking import class_logits, new_model, train_raw
from evaluation.latent_chunk_holdout import make_held_out_data
from evaluation.latent_chunk_reconstruction import LatentRoleTags, train_autoencoder
from evaluation.latent_chunk_relations import CLASSES, PHRASE_LENGTH
from evaluation.latent_chunk_soft_reader_confirmatory import (
    BITS,
    RANK,
    decode_int4,
    token_reconstruction_metrics,
    train_direct_int4,
    train_soft_reader,
)
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]
PHRASE_VOCAB = CLASSES + 2
SOURCE_PATHS = (
    "evaluation/latent_chunk_compact_reader.py",
    "evaluation/latent_chunk_soft_reader_confirmatory.py",
    "evaluation/latent_chunk_reconstruction.py",
    "evaluation/latent_chunk_ab_thinking.py",
    "evaluation/latent_chunk_holdout.py",
    "evaluation/latent_chunk_relations.py",
    "evaluation/provenance.py",
    "models/model_v2.py",
)


class CompactPairReader(nn.Module):
    """Merge adjacent decoded token embeddings into one vector per pair."""

    def __init__(self, d):
        super().__init__()
        self.merge = nn.Linear(2 * d, d)

    def forward(self, embeddings):
        batch, length, width = embeddings.shape
        expected_length = 2 * PHRASE_LENGTH
        if length != expected_length:
            raise ValueError(f"expected {expected_length} decoded vectors, got {length}")
        pairs = embeddings.reshape(batch, 2, PHRASE_LENGTH // 2, 2, width)
        return F.normalize(self.merge(pairs.flatten(-2)), dim=-1)


def compact_embeddings(model, source_model, encoder, reader, tags, ids):
    with torch.no_grad():
        a, b = encoder.encode(source_model, ids)
        a = (a * 7).round().clamp(-7, 7) / 7
        b = (b * 7).round().clamp(-7, 7) / 7
        probabilities = torch.cat((encoder.decode(a), encoder.decode(b)), dim=1).softmax(-1)

    token_embeddings = F.normalize(model.tok.weight[:PHRASE_VOCAB], dim=-1)
    decoded = probabilities @ token_embeddings
    merged = reader(decoded)
    return tags.apply(merged[:, 0], merged[:, 1])


@torch.no_grad()
def evaluate_compact_reader(model, source_model, encoder, reader, tags, data, batch_size=256):
    model.eval()
    source_model.eval()
    encoder.eval()
    reader.eval()
    tags.eval()
    predictions = []
    for start in range(0, data["target"].numel(), batch_size):
        sl = slice(start, min(start + batch_size, data["target"].numel()))
        embeds = compact_embeddings(
            model, source_model, encoder, reader, tags, data["ids"][sl]
        )
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


def train_compact_reader(seed, source_model, encoder, train, test, args, device, initial_state):
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
        embeds = compact_embeddings(model, source_model, encoder, reader, tags, ids)
        output = model(None, 1, inputs_embeds=embeds, sample_noise=False)
        objective = F.cross_entropy(class_logits(model, output), train["target"][indices])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    return {
        "test": evaluate_compact_reader(model, source_model, encoder, reader, tags, test),
        "train": evaluate_compact_reader(model, source_model, encoder, reader, tags, train),
        "input_vectors": 2 * RANK,
        "additional_reader_parameters": sum(parameter.numel() for parameter in reader.parameters()),
        "role_tag_parameters": sum(parameter.numel() for parameter in tags.parameters()),
        "trainable_model_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def tagged_soft_embeddings(model, source_model, encoder, tags, ids):
    with torch.no_grad():
        a, b = encoder.encode(source_model, ids)
        a = (a * 7).round().clamp(-7, 7) / 7
        b = (b * 7).round().clamp(-7, 7) / 7
        probabilities = torch.cat((encoder.decode(a), encoder.decode(b)), dim=1).softmax(-1)

    token_embeddings = F.normalize(model.tok.weight[:PHRASE_VOCAB], dim=-1)
    decoded = probabilities @ token_embeddings
    batch, _, width = decoded.shape
    paired_slots = decoded.reshape(batch, 2, RANK, 2, width)
    slot_tags = tags.role + tags.position
    tagged = paired_slots + slot_tags.unsqueeze(0).unsqueeze(-2)
    return tagged.reshape(batch, 2 * PHRASE_LENGTH, width)


@torch.no_grad()
def evaluate_tagged_soft_reader(model, source_model, encoder, tags, data, batch_size=256):
    model.eval()
    source_model.eval()
    encoder.eval()
    tags.eval()
    predictions = []
    for start in range(0, data["target"].numel(), batch_size):
        sl = slice(start, min(start + batch_size, data["target"].numel()))
        embeds = tagged_soft_embeddings(
            model, source_model, encoder, tags, data["ids"][sl]
        )
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


def train_tagged_soft_reader(seed, source_model, encoder, train, test, args, device, initial_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_state)
    tags = LatentRoleTags(RANK, model.d).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable.extend(tags.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    torch.manual_seed(seed * 1009 + 7)
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 19)
    model.train()
    tags.train()

    for _ in range(args.steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        ids = train["ids"][indices]
        embeds = tagged_soft_embeddings(model, source_model, encoder, tags, ids)
        output = model(None, 1, inputs_embeds=embeds, sample_noise=False)
        objective = F.cross_entropy(class_logits(model, output), train["target"][indices])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    return {
        "test": evaluate_tagged_soft_reader(model, source_model, encoder, tags, test),
        "train": evaluate_tagged_soft_reader(model, source_model, encoder, tags, train),
        "input_vectors": 2 * PHRASE_LENGTH,
        "additional_reader_parameters": 0,
        "role_tag_parameters": sum(parameter.numel() for parameter in tags.parameters()),
        "trainable_model_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(report):
    by_offsets = {}
    for run in report["runs"]:
        key = tuple(run["held_out_offsets"])
        by_offsets.setdefault(key, []).append(run)

    lines = [
        "# Compact Reader Evaluation",
        "",
        f"- Status: {str(report['progress']['complete']).lower()} ({len(report['runs'])} completed runs)",
        f"- Model seeds: {report['config']['seeds']}",
        f"- Training steps per arm: {report['config']['steps']}",
        "- Results are mean ± sample SD across model seeds on each fixed balanced split.",
        "",
        "| Held-out offsets | Input condition | Vectors | Test accuracy | Pair exact |",
        "|---|---|---:|---:|---:|",
    ]
    arm_names = {
        "raw_token_baseline": "Raw tokens",
        "direct_int4_latent": "Direct int4 a/b",
        "compact_8_vector_reader": "Compact soft reader",
        "soft_16_vector_reader": "Full soft reader",
        "tagged_soft_16_vector_reader": "Matched-tag soft reader",
    }
    for offsets, runs in sorted(by_offsets.items()):
        for arm, label in arm_names.items():
            if any(arm not in run for run in runs):
                continue
            metrics = [
                run[arm] if arm == "raw_token_baseline" else run[arm]["test"]
                for run in runs
            ]
            acc_mean, acc_sd = mean_sd([item["accuracy"] for item in metrics])
            pair_mean, pair_sd = mean_sd(
                [item["counterfactual_pair_exact"] for item in metrics]
            )
            vectors = 8 if arm in (
                "direct_int4_latent", "compact_8_vector_reader"
            ) else 16
            lines.append(
                f"| {list(offsets)} | {label} | {vectors} | "
                f"{100 * acc_mean:.2f} ± {100 * acc_sd:.2f}% | "
                f"{100 * pair_mean:.2f} ± {100 * pair_sd:.2f}% |"
            )
    lines.extend([
        "",
        "The rank-4 a/b code has 128 B of ideal int4 payload, while its dequantized FP32 training representation uses 1,024 B. The compact reader first materializes sixteen decoded token embeddings, then merges them to eight 32-dimensional FP32 trunk inputs (1,024 B). The full reader sends all sixteen to the trunk (2,048 B). These figures exclude shared encoder/decoder and reader parameters. The synthetic 16-token input can be represented in 8 B with ideal 4-bit packing, so the latent payload is not smaller than packed source IDs. The compact reader adds 2,080 merge parameters and 512 role/slot-tag parameters.",
        "",
        "The compact reader merges adjacent decoded token pairs with one shared linear layer, then adds trainable phrase-role and slot tags. It is trained with task loss only. Comparing the two 16-vector readers tests the effect of adding tags. Comparing the compact and matched-tag readers also changes pair mixing and normalization alongside trunk input length. This synthetic benchmark does not establish natural-language retrieval or actual packed-runtime memory and latency savings.",
        "",
    ])
    return "\n".join(lines)


def write_json_atomic(path, report):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def make_offsets_sets(offsets):
    sets = [[offset] for offset in offsets if offset not in (3, 5)]
    if 3 in offsets and 5 in offsets:
        sets.append([3, 5])
    else:
        sets.extend([[offset] for offset in (3, 5) if offset in offsets])
    return sets


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
        default="experiments/latent_chunk/compact_reader_matched_tags_data0_model0-4_steps800.json",
    )
    cli = parser.parse_args()
    if min(cli.steps, cli.pretrain_steps, cli.batch_size) <= 0:
        parser.error("steps, pretrain steps, and batch size must be positive")
    if not cli.seeds or len(set(cli.seeds)) != len(cli.seeds):
        parser.error("seeds must be nonempty and unique")
    if not cli.held_out_offsets or any(offset not in range(1, CLASSES) for offset in cli.held_out_offsets):
        parser.error("held-out offsets must be a nonempty list from 1 to 7")
    if len(set(cli.held_out_offsets)) != len(cli.held_out_offsets):
        parser.error("held-out offsets must not contain duplicates")

    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    args = SimpleNamespace(
        steps=cli.steps, pretrain_steps=cli.pretrain_steps,
        batch_size=cli.batch_size, lr=cli.lr,
    )
    output_path = ROOT / cli.output
    offsets_sets = make_offsets_sets(cli.held_out_offsets)
    config = vars(cli).copy()
    config.pop("resume")
    provenance = capture(SOURCE_PATHS)

    if output_path.exists():
        if not cli.resume:
            parser.error(f"output already exists; pass --resume to continue: {output_path}")
        report = json.loads(output_path.read_text(encoding="utf-8"))
        if report["config"] != config:
            parser.error("resume configuration does not match the existing result")
        if report["provenance"]["source_sha256_at_capture"] != provenance["source_sha256_at_capture"]:
            parser.error("resume source hashes do not match the existing result")
        if any("tagged_soft_16_vector_reader" not in run for run in report["runs"]):
            parser.error("resume result is missing the matched-tag condition")
    else:
        report = {
            "task": "compact_reader_for_two_phrase_relation_composition",
            "method": "Train raw, direct int4 a/b, an 8-vector compact soft reader, an untagged 16-vector soft reader, and a 16-vector soft reader with the same phrase/slot tags as the compact arm on matched initializations, splits, minibatch sequences, and task budgets.",
            "config": config,
            "storage": {
                "fake_int4_ab_payload_bytes": 2 * RANK * 32 * BITS // 8,
                "dequantized_ab_fp32_bytes": 2 * RANK * 32 * 4,
                "compact_expanded_fp32_bytes": 2 * RANK * 32 * 4,
                "full_reader_expanded_fp32_bytes": 2 * PHRASE_LENGTH * 32 * 4,
                "ideal_packed_synthetic_token_ids_bytes": (
                    2 * PHRASE_LENGTH * math.ceil(math.log2(CLASSES + 2)) // 8
                ),
                "shared_encoder_decoder_excluded": True,
            },
            "reader_parameters": {
                "compact_merge_parameters": 2 * 32 * 32 + 32,
                "compact_role_and_position_parameters": 4 * RANK * 32,
                "full_soft_reader_additional_parameters": 0,
                "matched_tag_soft_reader_additional_parameters": 4 * RANK * 32,
            },
            "limitations": "Synthetic task with frozen randomly initialized source embeddings. The compact merge, tags, token embeddings, and trunk receive task gradients, while the phrase encoder/decoder stay frozen after reconstruction pretraining. The compact path materializes sixteen decoded embeddings before merging to eight trunk inputs. FP32 tensors simulate int4 values. No packed runtime, latency, natural-language, or online memory-write test is performed.",
            "torch": torch.__version__,
            "provenance": provenance,
            "progress": {"complete": False},
            "runs": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_path, report)

    completed = {
        (tuple(run["held_out_offsets"]), run["seed"])
        for run in report["runs"]
    }
    for offsets in offsets_sets:
        train, test, held_out_pairs = make_held_out_data(
            cli.data_seed, device, held_out_offsets=tuple(offsets)
        )
        for seed in cli.seeds:
            run_key = (tuple(offsets), seed)
            if run_key in completed:
                continue
            print(f"offsets {offsets}, model seed {seed}: preparing matched conditions", flush=True)
            initial_model = new_model(seed, device)
            initial_state = copy.deepcopy(initial_model.state_dict())
            del initial_model
            raw_args = SimpleNamespace(steps=args.steps, batch_size=args.batch_size, lr=args.lr)
            raw = train_raw(seed, train, test, raw_args, device, initial_state)
            source_model, encoder, reconstruction = train_autoencoder(
                seed, RANK, train, test,
                SimpleNamespace(
                    steps=args.pretrain_steps, batch_size=args.batch_size, lr=args.lr
                ),
                device, initial_state,
            )
            direct = train_direct_int4(
                seed, source_model, encoder, train, test, args, device, initial_state
            )
            soft = train_soft_reader(
                seed, source_model, encoder, train, test, args, device, initial_state
            )
            compact = train_compact_reader(
                seed, source_model, encoder, train, test, args, device, initial_state
            )
            tagged_soft = train_tagged_soft_reader(
                seed, source_model, encoder, train, test, args, device, initial_state
            )
            decoded_test = decode_int4(source_model, encoder, test)
            record = {
                "held_out_offsets": offsets,
                "held_out_pairs": held_out_pairs,
                "seed": seed,
                "train_examples": train["target"].numel(),
                "test_examples": test["target"].numel(),
                "reconstruction_pretraining": reconstruction,
                "int4_test_reconstruction": token_reconstruction_metrics(
                    decoded_test["ids"], test["ids"]
                ),
                "raw_token_baseline": raw,
                "direct_int4_latent": direct,
                "compact_8_vector_reader": compact,
                "soft_16_vector_reader": soft,
                "tagged_soft_16_vector_reader": tagged_soft,
                "compact_expanded_vectors": 2 * RANK,
                "full_reader_expanded_vectors": 2 * PHRASE_LENGTH,
            }
            report["runs"].append(record)
            completed.add(run_key)
            report["progress"]["complete"] = len(completed) == len(offsets_sets) * len(cli.seeds)
            write_json_atomic(output_path, report)
            print(json.dumps(record, ensure_ascii=False), flush=True)
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
