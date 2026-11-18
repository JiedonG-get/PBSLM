"""Distill separate int4 phrase writers for the frozen PBSLM trunk."""
import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from evaluation.latent_chunk_ab_thinking import class_logits, new_model, train_raw
from evaluation.latent_chunk_compact_int4 import fake_int4_ste, pack_int4, unpack_int4
from evaluation.latent_chunk_compact_student import (
    compact_teacher_inputs,
    mean_sd,
    phrase_codes,
    quantize_int4,
    score_predictions,
    teacher_features,
    train_teacher,
    write_json_atomic,
)
from evaluation.latent_chunk_holdout import make_held_out_data
from evaluation.latent_chunk_reconstruction import train_autoencoder
from evaluation.latent_chunk_relations import CLASSES, PHRASE_LENGTH
from evaluation.latent_chunk_soft_reader_confirmatory import (
    RANK,
    decode_int4,
    token_reconstruction_metrics,
    train_direct_int4,
)
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]
HELD_OUT_OFFSETS = (1, 7)
CONDITIONS = (
    ("raw_token_baseline", "Raw tokens"),
    ("direct_int4", "Direct int4 a/b with its own trunk"),
    ("direct_int4_frozen_teacher", "Direct int4 a/b in the frozen teacher"),
    ("teacher_float", "Decoder-mediated teacher, FP32 trunk input"),
    ("teacher_packed", "Decoder-mediated teacher, packed int4 trunk input"),
    ("feature_only", "Packed writer, feature loss"),
    ("feature_and_task", "Packed writer, feature + task losses"),
)
SOURCE_PATHS = (
    "evaluation/latent_chunk_compact_writer.py",
    "evaluation/latent_chunk_ab_thinking.py",
    "evaluation/latent_chunk_compact_int4.py",
    "evaluation/latent_chunk_compact_reader.py",
    "evaluation/latent_chunk_compact_student.py",
    "evaluation/latent_chunk_holdout.py",
    "evaluation/latent_chunk_reconstruction.py",
    "evaluation/latent_chunk_relations.py",
    "evaluation/latent_chunk_soft_reader_confirmatory.py",
    "evaluation/provenance.py",
    "models/model_v2.py",
)


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def write_codes(writer, source_model, ids):
    """Read each eight-token segment with the same position-sensitive encoder."""
    phrases = ids.reshape(ids.shape[0], 2, PHRASE_LENGTH)
    with torch.no_grad():
        embeddings = source_model.token_emb(phrases)
    codes = writer(embeddings.flatten(-2)).unflatten(-1, (RANK, source_model.d))
    return F.normalize(codes, dim=-1)


def packed_roundtrip(values):
    packed = pack_int4(values)
    restored = unpack_int4(packed, values.shape)
    if packed.dtype != torch.uint8 or packed.numel() * 2 != values.numel():
        raise AssertionError("int4 storage must contain exactly two values per byte")
    if not torch.equal(restored, quantize_int4(values)):
        raise AssertionError("int4 packing changed quantized values")
    return restored, packed.numel()


@torch.no_grad()
def evaluate_teacher_conditions(teacher, source_model, encoder, data, batch_size=256):
    keys = ("teacher_float", "teacher_packed", "direct_int4_frozen_teacher")
    predictions = {key: [] for key in keys}
    payload = {key: 0 for key in keys}
    for start in range(0, data["target"].numel(), batch_size):
        ids = data["ids"][start:start + batch_size]
        codes, features = teacher_features(teacher, source_model, encoder, ids)
        tagged = compact_teacher_inputs(teacher, features)
        packed_teacher, teacher_bytes = packed_roundtrip(tagged)
        tagged_direct = compact_teacher_inputs(teacher, codes)
        packed_direct, direct_bytes = packed_roundtrip(tagged_direct)
        payload["teacher_float"] += tagged.numel() * tagged.element_size()
        payload["teacher_packed"] += teacher_bytes
        payload["direct_int4_frozen_teacher"] += direct_bytes
        for key, inputs in zip(keys, (tagged, packed_teacher, packed_direct)):
            output = teacher["model"](None, 1, inputs_embeds=inputs, sample_noise=False)
            predictions[key].append(class_logits(teacher["model"], output).argmax(-1))
    return {
        key: {
            **score_predictions(torch.cat(predictions[key]), data["target"]),
            "trunk_input_bytes_per_example": payload[key] / data["target"].numel(),
        }
        for key in keys
    }


@torch.no_grad()
def evaluate_writer(writer, teacher, source_model, encoder, data, batch_size=256):
    writer.eval()
    predictions = []
    stored_bytes = 0
    tagged_bytes = 0
    cosine_total = torch.zeros((), device=data["ids"].device)
    for start in range(0, data["target"].numel(), batch_size):
        ids = data["ids"][start:start + batch_size]
        features = write_codes(writer, source_model, ids)
        restored, batch_stored_bytes = packed_roundtrip(features)
        tagged = compact_teacher_inputs(teacher, restored)
        inputs, batch_tagged_bytes = packed_roundtrip(tagged)
        stored_bytes += batch_stored_bytes
        tagged_bytes += batch_tagged_bytes

        output = teacher["model"](None, 1, inputs_embeds=inputs, sample_noise=False)
        predictions.append(class_logits(teacher["model"], output).argmax(-1))

        _, targets = teacher_features(teacher, source_model, encoder, ids)
        cosine_total += F.cosine_similarity(restored, targets, dim=-1).sum()

    size = data["target"].numel()
    return {
        **score_predictions(torch.cat(predictions), data["target"]),
        "stored_ab_code_bytes_per_example": stored_bytes / size,
        "trunk_input_bytes_per_example": tagged_bytes / size,
        "stored_code_teacher_feature_cosine": float(cosine_total) / (size * 2 * RANK),
    }


def train_writer(condition, seed, teacher, source_model, encoder, train, test, targets, args):
    if any(parameter.requires_grad for module in teacher.values() for parameter in module.parameters()):
        raise AssertionError("teacher weights must be frozen during writer training")
    writer = copy.deepcopy(encoder.encoder).requires_grad_(True)
    writer.train()
    optimizer = torch.optim.AdamW(writer.parameters(), lr=args.lr, weight_decay=1e-4)
    torch.manual_seed(seed * 1009 + 91)
    rng = torch.Generator(device=train["target"].device).manual_seed(seed * 997 + 29)
    use_task = condition == "feature_and_task"

    for step in range(args.steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=train["target"].device
        )
        codes = write_codes(writer, source_model, train["ids"][indices])
        feature_loss = (1 - (codes * targets[indices]).sum(-1)).mean()
        objective = feature_loss
        if use_task:
            tagged = compact_teacher_inputs(teacher, fake_int4_ste(codes))
            output = teacher["model"](None, 1, inputs_embeds=fake_int4_ste(tagged), sample_noise=False)
            task_loss = F.cross_entropy(class_logits(teacher["model"], output), train["target"][indices])
            objective = objective + task_loss
            if step == 0:
                task_gradient, = torch.autograd.grad(task_loss, codes, retain_graph=True)
                if not bool(torch.isfinite(task_gradient).all() and torch.any(task_gradient != 0)):
                    raise AssertionError("task loss did not produce a finite nonzero gradient to writer codes")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0)
        optimizer.step()

    return {
        "test": evaluate_writer(writer, teacher, source_model, encoder, test),
        "train": evaluate_writer(writer, teacher, source_model, encoder, train),
        "trainable_writer_parameters": parameter_count(writer),
    }


def run_one(seed, train, test, args, device):
    initial_model = new_model(seed, device)
    initial_state = initial_model.state_dict()
    trunk_parameters = parameter_count(initial_model)
    trunk_trainable = sum(p.numel() for p in initial_model.parameters() if p.requires_grad)
    raw = train_raw(seed, train, test, args, device, initial_state)
    pretrain_args = SimpleNamespace(steps=args.pretrain_steps, batch_size=args.batch_size, lr=args.lr)
    source_model, encoder, reconstruction = train_autoencoder(
        seed, RANK, train, test, pretrain_args, device, initial_state
    )
    direct = train_direct_int4(seed, source_model, encoder, train, test, args, device, initial_state)
    teacher = train_teacher(seed, source_model, encoder, train, args, device, initial_state)

    with torch.no_grad():
        sample_ids = train["ids"][:16]
        initial_codes = write_codes(encoder.encoder, source_model, sample_ids)
        packed_initial, _ = packed_roundtrip(initial_codes)
        torch.testing.assert_close(packed_initial, phrase_codes(source_model, encoder, sample_ids))
        swapped_ids = sample_ids.reshape(-1, 2, PHRASE_LENGTH).flip(1).flatten(1)
        swapped_codes = write_codes(encoder.encoder, source_model, swapped_ids)
        torch.testing.assert_close(initial_codes.flip(1), swapped_codes)
        _, training_targets = teacher_features(teacher, source_model, encoder, train["ids"])
        int4_reconstruction = token_reconstruction_metrics(
            decode_int4(source_model, encoder, test)["ids"], test["ids"]
        )

    teacher_test = evaluate_teacher_conditions(teacher, source_model, encoder, test)
    teacher_train = evaluate_teacher_conditions(teacher, source_model, encoder, train)
    record = {
        "seed": seed,
        "train_examples": train["target"].numel(),
        "test_examples": test["target"].numel(),
        "parameters": {
            "trunk": trunk_parameters,
            "trunk_trainable_in_baselines": trunk_trainable,
            "frozen_source_embedding_table": parameter_count(source_model.tok),
            "phrase_encoder_or_writer": parameter_count(encoder.encoder),
            "phrase_decoder": parameter_count(encoder.decoder),
            "teacher_merge": parameter_count(teacher["reader"]),
            "role_and_position_tags": parameter_count(teacher["tags"]),
        },
        "reconstruction_pretraining": reconstruction,
        "source_int4_test_reconstruction": int4_reconstruction,
        "raw_token_baseline": {"test": raw, "train": raw["train_evaluation"]},
        "direct_int4": direct,
        **{key: {"test": teacher_test[key], "train": teacher_train[key]} for key in teacher_test},
    }
    for condition in ("feature_only", "feature_and_task"):
        record[condition] = train_writer(
            condition, seed, teacher, source_model, encoder, train, test, training_targets, args
        )
    return record


def summarize(report):
    lines = [
        "# Decoder-Free Compact Writer",
        "",
        f"Held-out modular offsets {report['held_out_offsets']}, data seed {report['config']['data_seed']}.",
        f"Completed {len(report['runs'])} of {len(report['config']['seeds'])} model seeds.",
        "The table shows mean +/- sample SD across model seeds on this fixed split.",
        "",
        "| Condition | Test accuracy | Counterfactual pair exact |",
        "|---|---:|---:|",
    ]
    for key, label in CONDITIONS:
        metrics = [run[key]["test"] for run in report["runs"]]
        accuracy, accuracy_sd = mean_sd([result["accuracy"] for result in metrics])
        pair_exact, pair_sd = mean_sd([result["counterfactual_pair_exact"] for result in metrics])
        lines.append(
            f"| {label} | {100 * accuracy:.2f} +/- {100 * accuracy_sd:.2f}% | "
            f"{100 * pair_exact:.2f} +/- {100 * pair_sd:.2f}% |"
        )
    lines.extend([
        "",
        "Each writer starts from the same reconstruction-pretrained phrase encoder. Each phrase produces four "
        "32-dimensional vectors from frozen token embeddings. Only training rows supply reconstruction updates, "
        "teacher supervision, distillation targets, and writer updates. The teacher stays frozen during distillation. "
        "The feature loss matches the pre-tag merged vectors. The joint arm gives task cross-entropy and feature loss "
        "equal weight and uses the same starting weights and minibatches as the feature-only arm.",
        "",
        "The writer stores each pre-tag phrase code in 64 bytes, or 128 bytes for both phrases. It unpacks those codes, "
        "adds learned phrase/position tags, then packs and unpacks a separate 128-byte int4 trunk-input payload. "
        "The trunk computes in FP32 after unpacking. Ideal packing of the 16 synthetic token IDs uses 8 bytes. "
        "Writer accuracy is measured on a synthetic task with two simultaneously available phrases.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--teacher-steps", type=int, default=800)
    parser.add_argument("--pretrain-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output", default="experiments/latent_chunk/compact_writer_balanced_data0_offsets1-7_model0-4_steps800.json"
    )
    args = parser.parse_args()
    if min(args.pretrain_steps, args.teacher_steps, args.steps, args.batch_size) <= 0:
        parser.error("training steps and batch size must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("model seeds must be unique")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("learning rate must be finite and positive")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    output = ROOT / args.output
    config = {key: value for key, value in vars(args).items() if key != "resume"}
    provenance = capture(SOURCE_PATHS)
    train, test, held_out_pairs = make_held_out_data(args.data_seed, device, held_out_offsets=HELD_OUT_OFFSETS)

    if output.exists():
        if not args.resume:
            parser.error(f"output exists; pass --resume to continue: {output}")
        report = json.loads(output.read_text(encoding="utf-8"))
        if report["config"] != config:
            parser.error("resume configuration does not match the existing result")
        if report["provenance"]["source_sha256_at_capture"] != provenance["source_sha256_at_capture"]:
            parser.error("resume source hashes do not match the existing result")
    else:
        report = {
            "task": "decoder_free_packed_int4_phrase_writer",
            "config": config,
            "held_out_offsets": list(HELD_OUT_OFFSETS),
            "held_out_pairs": held_out_pairs,
            "storage": {
                "source_autoencoder_code_bytes_per_example": 2 * RANK * 32 // 2,
                "writer_persistent_code_bytes_per_phrase": RANK * 32 // 2,
                "writer_persistent_ab_code_bytes_per_example": 2 * RANK * 32 // 2,
                "writer_temporary_packed_trunk_input_bytes_per_example": 2 * RANK * 32 // 2,
                "unpacked_trunk_input_bytes_per_example": 2 * RANK * 32 * 4,
                "ideal_packed_synthetic_token_ids_bytes_per_example": (
                    2 * PHRASE_LENGTH * (CLASSES + 1).bit_length() // 8
                ),
            },
            "storage_notes": "Payload sizes exclude model weights and runtime buffers. The writer persists only pre-tag a/b codes. Trunk inputs are unpacked to FP32 for computation.",
            "training_notes": "Raw and direct baselines train their own trunks on training labels. The decoder-mediated teacher trains on training labels with unquantized tagged trunk inputs. Writer conditions use the same phrase-encoder initialization, training rows, and minibatches, with a frozen teacher trunk and frozen source embeddings.",
            "provenance": provenance,
            "progress": {"complete": False},
            "runs": [],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output, report)

    completed = {run["seed"] for run in report["runs"]}
    for seed in args.seeds:
        if seed in completed:
            continue
        print(f"held-out offsets {HELD_OUT_OFFSETS}, model seed {seed}: training compact writer conditions", flush=True)
        record = run_one(seed, train, test, args, device)
        report["runs"].append(record)
        completed.add(seed)
        report["progress"]["complete"] = completed == set(args.seeds)
        write_json_atomic(output, report)
        metrics = {key: record[key]["test"]["accuracy"] for key, _ in CONDITIONS}
        print(json.dumps({"seed": seed, "test_accuracy": metrics}), flush=True)

    output.with_suffix(".md").write_text(summarize(report), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
