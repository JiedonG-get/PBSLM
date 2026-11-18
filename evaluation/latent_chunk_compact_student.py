"""Distill a vocabulary-decoder-free reader for int4 phrase codes."""
import argparse
import copy
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.latent_chunk_ab_thinking import class_logits, new_model, train_raw
from evaluation.latent_chunk_compact_int4 import (
    fake_int4_ste,
    make_offsets_sets,
    pack_int4,
    unpack_int4,
)
from evaluation.latent_chunk_compact_reader import CompactPairReader
from evaluation.latent_chunk_holdout import make_held_out_data
from evaluation.latent_chunk_reconstruction import LatentRoleTags, train_autoencoder
from evaluation.latent_chunk_relations import CLASSES, PHRASE_LENGTH
from evaluation.latent_chunk_soft_reader_confirmatory import RANK
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]
PHRASE_VOCAB = CLASSES + 2
CODE_WIDTH = RANK * 32


def quantize_int4(values):
    return (values * 7).round().clamp(-7, 7) / 7


class LatentToCompact(nn.Module):
    """Map one quantized phrase code directly into four trunk vectors."""

    def __init__(self, d):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(RANK * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, RANK * d),
        )

    def forward(self, codes):
        batch, phrases, rank, width = codes.shape
        vectors = self.network(codes.reshape(batch * phrases, rank * width))
        vectors = vectors.reshape(batch, phrases, rank, width)
        return F.normalize(vectors, dim=-1)


def phrase_codes(source_model, encoder, ids):
    with torch.no_grad():
        a, b = encoder.encode(source_model, ids)
        codes = torch.stack((quantize_int4(a), quantize_int4(b)), dim=1)
        packed = pack_int4(codes)
        return unpack_int4(packed, codes.shape)


def teacher_features(teacher, source_model, encoder, ids):
    codes = phrase_codes(source_model, encoder, ids)
    with torch.no_grad():
        distributions = torch.cat(
            (teacher["encoder"].decode(codes[:, 0]), teacher["encoder"].decode(codes[:, 1])),
            dim=1,
        ).softmax(-1)
    token_embeddings = F.normalize(teacher["model"].tok.weight[:PHRASE_VOCAB], dim=-1)
    decoded = distributions @ token_embeddings
    merged = teacher["reader"](decoded)
    features = merged.reshape(ids.shape[0], 2, RANK, -1)
    return codes, features


def compact_teacher_inputs(teacher, features):
    return teacher["tags"].apply(features[:, 0], features[:, 1])


def evaluate_teacher(teacher, source_model, encoder, data, batch_size=256):
    model = teacher["model"]
    model.eval()
    source_model.eval()
    encoder.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, data["target"].numel(), batch_size):
            sl = slice(start, min(start + batch_size, data["target"].numel()))
            _, features = teacher_features(
                teacher, source_model, encoder, data["ids"][sl]
            )
            output = model(
                None, 1,
                inputs_embeds=compact_teacher_inputs(teacher, features),
                sample_noise=False,
            )
            predictions.extend(class_logits(model, output).argmax(-1).tolist())
    return score_predictions(torch.tensor(predictions, device=data["target"].device), data["target"])


def score_predictions(predictions, targets):
    pair_exact = (
        predictions[0::2].eq(targets[0::2])
        & predictions[1::2].eq(targets[1::2])
    ).float().mean()
    return {
        "accuracy": float(predictions.eq(targets).float().mean()),
        "counterfactual_pair_exact": float(pair_exact),
        "examples": targets.numel(),
    }


def train_teacher(seed, source_model, encoder, train, args, device, initial_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_state)
    reader = CompactPairReader(model.d).to(device)
    tags = LatentRoleTags(RANK, model.d).to(device)
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
    teacher = {"model": model, "reader": reader, "tags": tags, "encoder": encoder}

    for _ in range(args.teacher_steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        _, features = teacher_features(teacher, source_model, encoder, train["ids"][indices])
        output = model(
            None, 1,
            inputs_embeds=compact_teacher_inputs(teacher, features),
            sample_noise=False,
        )
        objective = F.cross_entropy(class_logits(model, output), train["target"][indices])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    for module in (model, reader, tags, encoder):
        module.eval()
    for module in (model, reader, tags, encoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def evaluate_direct_codes(teacher, codes, data, batch_size=256):
    predictions = []
    model = teacher["model"]
    model.eval()
    for start in range(0, data["target"].numel(), batch_size):
        sl = slice(start, min(start + batch_size, data["target"].numel()))
        embeds = teacher["tags"].apply(codes[sl, 0], codes[sl, 1])
        output = model(None, 1, inputs_embeds=embeds, sample_noise=False)
        predictions.extend(class_logits(model, output).argmax(-1).tolist())
    return score_predictions(torch.tensor(predictions, device=data["target"].device), data["target"])


@torch.no_grad()
def evaluate_student(teacher, adapter, codes, data, batch_size=256):
    predictions = []
    packed_bytes = 0
    adapter.eval()
    model = teacher["model"]
    for start in range(0, data["target"].numel(), batch_size):
        sl = slice(start, min(start + batch_size, data["target"].numel()))
        features = adapter(codes[sl])
        tagged = teacher["tags"].apply(features[:, 0], features[:, 1])
        packed = pack_int4(tagged)
        packed_bytes += packed.numel()
        stored_vectors = unpack_int4(packed, tagged.shape)
        output = model(None, 1, inputs_embeds=stored_vectors, sample_noise=False)
        predictions.extend(class_logits(model, output).argmax(-1).tolist())
    result = score_predictions(
        torch.tensor(predictions, device=data["target"].device), data["target"]
    )
    result["packed_memory_bytes_per_example"] = packed_bytes / data["target"].numel()
    return result


def train_student_condition(
    condition, seed, teacher, train, train_codes, train_features,
    test, test_codes, args, device, initial_adapter,
):
    adapter = LatentToCompact(32).to(device)
    adapter.load_state_dict(initial_adapter)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    torch.manual_seed(seed * 1009 + 91)
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 29)
    adapter.train()

    for _ in range(args.steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        features = adapter(train_codes[indices])
        objective = features.new_zeros(())
        if condition in ("feature_only", "feature_and_task"):
            cosine = (features * train_features[indices]).sum(-1)
            objective = objective + (1.0 - cosine).mean()
        if condition in ("task_only", "feature_and_task"):
            tagged = teacher["tags"].apply(features[:, 0], features[:, 1])
            output = teacher["model"](
                None, 1,
                inputs_embeds=fake_int4_ste(tagged),
                sample_noise=False,
            )
            task_loss = F.cross_entropy(
                class_logits(teacher["model"], output), train["target"][indices]
            )
            objective = objective + task_loss
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

    result = {
        "test": evaluate_student(teacher, adapter, test_codes, test),
        "train": evaluate_student(teacher, adapter, train_codes, train),
        "trainable_adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "condition": condition,
    }
    return result


def mean_sd(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def summarize(report):
    lines = [
        "# Decoder-Free Compact Reader",
        "",
        f"- Status: {str(report['progress']['complete']).lower()} ({len(report['runs'])} completed runs)",
        f"- Model seeds: {report['config']['seeds']}",
        f"- Student task steps: {report['config']['steps']}",
        "- Results are mean ± sample SD across model seeds on each fixed balanced split.",
        "",
        "| Held-out offsets | Input condition | Test accuracy | Pair exact |",
        "|---|---|---:|---:|",
    ]
    arms = [
        ("raw_token_baseline", "Raw-token baseline"),
        ("compact_teacher", "Decoder-mediated compact teacher"),
        ("direct_int4_codes", "Direct int4 `a/b` codes"),
        ("feature_only", "Feature-distilled int4 adapter"),
        ("task_only", "Task-trained int4 adapter"),
        ("feature_and_task", "Feature + task int4 adapter"),
    ]
    offsets_sets = sorted({tuple(run["held_out_offsets"]) for run in report["runs"]})
    for offsets in offsets_sets:
        runs = [run for run in report["runs"] if tuple(run["held_out_offsets"]) == offsets]
        for key, label in arms:
            metrics = [
                run[key] if key == "raw_token_baseline" else run[key]["test"]
                for run in runs
            ]
            acc_mean, acc_sd = mean_sd([item["accuracy"] for item in metrics])
            pair_mean, pair_sd = mean_sd([item["counterfactual_pair_exact"] for item in metrics])
            lines.append(
                f"| {list(offsets)} | {label} | {100 * acc_mean:.2f} ± {100 * acc_sd:.2f}% | "
                f"{100 * pair_mean:.2f} ± {100 * pair_sd:.2f}% |"
            )
    lines.extend([
        "",
        "The student reads only the stored rank-4 int4 `a/b` vectors. A shared 128→64→128 MLP maps each phrase code to four compact trunk vectors. Feature-distillation targets come from the compact teacher's pre-tag merger outputs on training examples. Feature-only, task-only, and combined objectives are compared with matched adapter initialization and minibatch draws. The compact teacher uses the phrase decoder during reading, while student evaluation does not.",
        "",
        "Evaluation packs the student's eight tagged 32-dimensional vectors into 128 bytes and unpacks them before the frozen PBSLM trunk. The adapter adds 16,576 shared parameters. Source-phrase encoding still requires the existing encoder when writing a memory. This is a synthetic relation-composition benchmark and does not demonstrate language generalization, end-to-end storage savings, or online memory writing.",
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
    parser.add_argument("--teacher-steps", type=int, default=800)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--pretrain-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        default="experiments/latent_chunk/compact_reader_decoder_free_data0_model0-4_steps800.json",
    )
    cli = parser.parse_args()
    if min(cli.steps, cli.teacher_steps, cli.pretrain_steps, cli.batch_size) <= 0:
        parser.error("training steps and batch size must be positive")
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
        steps=cli.steps, teacher_steps=cli.teacher_steps,
        pretrain_steps=cli.pretrain_steps, batch_size=cli.batch_size, lr=cli.lr,
    )
    output_path = ROOT / cli.output
    offsets_sets = [tuple(offsets) for offsets in make_offsets_sets(cli.held_out_offsets)]
    config = vars(cli).copy()
    config.pop("resume")
    source_paths = [
        "evaluation/latent_chunk_compact_student.py",
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
        required = {
            "raw_token_baseline", "compact_teacher", "direct_int4_codes",
            "feature_only", "task_only", "feature_and_task",
        }
        if any(not required.issubset(run) for run in report["runs"]):
            parser.error("resume result is missing a required condition")
    else:
        report = {
            "task": "decoder_free_int4_compact_latent_reader",
            "method": "Distill a student from quantized phrase codes to the frozen compact reader features, comparing feature-only, task-only, and combined losses. At inference the student receives stored a/b codes and does not call the phrase decoder.",
            "config": config,
            "storage": {
                "source_ab_code_bytes_per_example": 2 * CODE_WIDTH // 2,
                "student_output_bytes_per_example": 2 * CODE_WIDTH // 2,
                "ideal_packed_synthetic_token_ids_bytes": 8,
                "student_adapter_parameters": (CODE_WIDTH * 64 + 64) + (64 * CODE_WIDTH + CODE_WIDTH),
            },
            "limitations": "Synthetic held-out relation task. The teacher and its decoder train only on training rows. Student evaluation uses stored int4 a/b codes through a learned adapter and a frozen teacher trunk, without phrase decoding. This does not demonstrate natural-language generalization or end-to-end storage savings.",
            "provenance": provenance,
            "progress": {"complete": False},
            "runs": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_path, report)

    completed = {
        (tuple(run["held_out_offsets"]), run["seed"]) for run in report["runs"]
    }
    for offsets in offsets_sets:
        train, test, held_out_pairs = make_held_out_data(
            cli.data_seed, device, held_out_offsets=offsets
        )
        for seed in cli.seeds:
            run_key = (offsets, seed)
            if run_key in completed:
                continue
            print(f"offsets {offsets}, model seed {seed}: preparing decoder-free conditions", flush=True)
            initial_model = new_model(seed, device)
            initial_state = copy.deepcopy(initial_model.state_dict())
            del initial_model
            raw = train_raw(seed, train, test, args, device, initial_state)
            source_model, encoder, reconstruction = train_autoencoder(
                seed, RANK, train, test,
                SimpleNamespace(steps=cli.pretrain_steps, batch_size=cli.batch_size, lr=cli.lr),
                device, initial_state,
            )
            teacher = train_teacher(seed, source_model, encoder, train, args, device, initial_state)
            with torch.no_grad():
                train_codes, train_features = teacher_features(
                    teacher, source_model, encoder, train["ids"]
                )
            test_codes = phrase_codes(source_model, encoder, test["ids"])
            direct = evaluate_direct_codes(teacher, test_codes, test)
            teacher_test = evaluate_teacher(teacher, source_model, encoder, test)
            initial_adapter_model = LatentToCompact(32).to(device)
            initial_adapter = copy.deepcopy(initial_adapter_model.state_dict())
            del initial_adapter_model
            students = {}
            for condition in ("feature_only", "task_only", "feature_and_task"):
                students[condition] = train_student_condition(
                    condition, seed, teacher, train, train_codes, train_features,
                    test, test_codes, args, device, initial_adapter,
                )
            record = {
                "held_out_offsets": list(offsets),
                "held_out_pairs": held_out_pairs,
                "seed": seed,
                "train_examples": train["target"].numel(),
                "test_examples": test["target"].numel(),
                "reconstruction_pretraining": reconstruction,
                "raw_token_baseline": raw,
                "compact_teacher": {"test": teacher_test, "decoder_used_at_inference": True},
                "direct_int4_codes": {"test": direct, "decoder_used_at_inference": False},
                **students,
                "student_adapter_parameters": sum(p.numel() for p in initial_adapter.values()),
                "packed_student_code_bytes_per_example": 2 * CODE_WIDTH // 2,
            }
            report["runs"].append(record)
            completed.add(run_key)
            report["progress"]["complete"] = len(completed) == len(offsets_sets) * len(cli.seeds)
            write_json_atomic(output_path, report)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            del source_model, encoder, teacher
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
