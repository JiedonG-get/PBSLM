"""Repeat the joint-int4 comparison with both sets of role tags trainable.

The historical frozen control applied its role tags under no_grad. This entry
point keeps that runner available to reproduce the pilot and matches the tag
training in the frozen and joint conditions.
"""

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluation.latent_chunk_task_aware_joint_reconstruction import (
    BITS,
    CLASSES,
    PHRASE_LENGTH,
    RANK,
    RECONSTRUCTION_WEIGHT,
    ROOT,
    LatentRoleTags,
    artifact_sha256,
    atomic_write_text,
    atomic_write_json,
    capture,
    class_logits,
    evaluate_int4_reconstruction,
    evaluate_task,
    new_model,
    quantize_int4,
    quantize_int4_ste,
    train_autoencoder,
    train_joint,
    train_raw_with_pair_metrics,
    write_markdown as write_joint_markdown,
)
from evaluation.latent_chunk_holdout import make_held_out_data


def write_markdown(report, path):
    write_joint_markdown(report, path)
    markdown = path.read_text(encoding="utf-8")
    offsets = report["split"]["held_out_offsets"]
    markdown = markdown.replace(
        "offset-4 ordered value pairs held out before training",
        f"ordered value pairs at offsets {offsets} held out before training",
    )
    path.write_text(markdown, encoding="utf-8")


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
    initial_tags = [parameter.detach().clone() for parameter in tags.parameters()]
    started = time.perf_counter()

    for _ in range(args.frozen_steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        with torch.no_grad():
            a, b = encoder.encode(source_model, train["ids"][indices])
            a, b = quantize_int4(a), quantize_int4(b)
        output = model(None, 1, inputs_embeds=tags.apply(a, b), sample_noise=False)
        objective = F.cross_entropy(class_logits(model, output), train["target"][indices])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    role_tag_updates = {
        name: float((parameter.detach() - initial).abs().max())
        for (name, parameter), initial in zip(tags.named_parameters(), initial_tags, strict=True)
    }
    if not all(math.isfinite(update) and update > 0.0 for update in role_tag_updates.values()):
        raise RuntimeError(f"Frozen-control role tags did not update as expected: {role_tag_updates}")

    return {
        "train": evaluate_task(model, source_model, encoder, tags, train),
        "test": evaluate_task(model, source_model, encoder, tags, test),
        "final_batch_task_loss": float(objective.detach()),
        "train_seconds": time.perf_counter() - started,
        "trainable_trunk_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "trainable_role_tag_parameters": sum(parameter.numel() for parameter in tags.parameters()),
        "max_abs_role_tag_parameter_updates": role_tag_updates,
    }


def train_task_only_encoder(seed, source_model, encoder, train, test, args, device, initial_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_state)
    tags = LatentRoleTags(RANK, model.d).to(device)
    for parameter in encoder.decoder.parameters():
        parameter.requires_grad_(False)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable.extend(encoder.encoder.parameters())
    trainable.extend(tags.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    torch.manual_seed(seed * 1009 + 7)
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 19)
    model.train()
    encoder.train()
    tags.train()
    initial_encoder = [parameter.detach().clone() for parameter in encoder.encoder.parameters()]
    started = time.perf_counter()

    for _ in range(args.joint_steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        ids = train["ids"][indices]
        a, b = encoder.encode(source_model, ids)
        embeds = tags.apply(quantize_int4_ste(a), quantize_int4_ste(b))
        output = model(None, 1, inputs_embeds=embeds, sample_noise=False)
        objective = F.cross_entropy(class_logits(model, output), train["target"][indices])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    encoder_updates = [
        float((parameter.detach() - initial).abs().max())
        for parameter, initial in zip(encoder.encoder.parameters(), initial_encoder, strict=True)
    ]
    if not all(math.isfinite(update) and update > 0.0 for update in encoder_updates):
        raise RuntimeError(f"Task-only encoder did not update as expected: {encoder_updates}")

    return {
        "train": evaluate_task(model, source_model, encoder, tags, train),
        "test": evaluate_task(model, source_model, encoder, tags, test),
        "int4_train_reconstruction": evaluate_int4_reconstruction(source_model, encoder, train),
        "int4_test_reconstruction": evaluate_int4_reconstruction(source_model, encoder, test),
        "final_batch_task_loss": float(objective.detach()),
        "train_seconds": time.perf_counter() - started,
        "trainable_trunk_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "trainable_encoder_parameters": sum(
            parameter.numel() for parameter in encoder.encoder.parameters()
        ),
        "trainable_role_tag_parameters": sum(parameter.numel() for parameter in tags.parameters()),
        "encoder_parameter_max_abs_updates": max(encoder_updates),
    }


def run_seed(seed, train, test, args, device):
    initial_model = new_model(seed, device)
    initial_state = copy.deepcopy(initial_model.state_dict())
    del initial_model

    print(f"model seed {seed}: training raw-token baseline", flush=True)
    raw_args = argparse.Namespace(steps=args.baseline_steps, batch_size=args.batch_size, lr=args.lr)
    raw = train_raw_with_pair_metrics(seed, train, test, raw_args, device, initial_state)

    print(f"model seed {seed}: pretraining phrase autoencoder", flush=True)
    pretrain_args = argparse.Namespace(
        steps=args.pretrain_steps, batch_size=args.batch_size, lr=args.lr
    )
    source_model, encoder, pretraining = train_autoencoder(
        seed, RANK, train, test, pretrain_args, device, initial_state
    )
    pretraining_int4 = {
        "train": evaluate_int4_reconstruction(source_model, encoder, train),
        "test": evaluate_int4_reconstruction(source_model, encoder, test),
    }

    print(f"model seed {seed}: training frozen int4 control with trainable tags", flush=True)
    frozen = train_frozen_int4(seed, source_model, encoder, train, test, args, device, initial_state)

    print(f"model seed {seed}: adapting int4 encoder with task loss only", flush=True)
    task_only_encoder = copy.deepcopy(encoder)
    task_only = train_task_only_encoder(
        seed, source_model, task_only_encoder, train, test, args, device, initial_state
    )

    print(f"model seed {seed}: jointly training int4 encoder, decoder and trunk", flush=True)
    joint = train_joint(seed, source_model, encoder, train, test, args, device, initial_state)
    return {
        "model_seed": seed,
        "train_examples": int(train["target"].numel()),
        "test_examples": int(test["target"].numel()),
        "autoencoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "shared_encoder_parameters": sum(parameter.numel() for parameter in encoder.encoder.parameters()),
        "shared_decoder_parameters": sum(parameter.numel() for parameter in encoder.decoder.parameters()),
        "pretraining": pretraining,
        "pretraining_int4_reconstruction": pretraining_int4,
        "frozen_int4_latent": frozen,
        "task_only_adapted_int4_latent": task_only,
        "joint_int4_latent": joint,
        "raw_token_baseline": raw,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--held-out-offsets", nargs="+", type=int, default=[4])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--pretrain-steps", type=int, default=800)
    parser.add_argument("--frozen-steps", type=int, default=1600)
    parser.add_argument("--joint-steps", type=int, default=1600)
    parser.add_argument("--baseline-steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(set(args.model_seeds)) != len(args.model_seeds):
        parser.error("--model-seeds must not contain duplicates")
    if min(args.pretrain_steps, args.frozen_steps, args.joint_steps, args.baseline_steps,
           args.batch_size) <= 0:
        parser.error("training steps and batch size must be positive")
    output_path = ROOT / Path(args.output)
    checkpoint_dir = output_path.parent / f"{output_path.stem}_checkpoints"
    if output_path.suffix != ".json":
        parser.error("--output must end in .json")
    if any(path.exists() for path in (output_path, output_path.with_suffix(".md"), checkpoint_dir)):
        parser.error("output or checkpoint already exists; choose a new output path")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    train, test, held_out_pairs = make_held_out_data(
        args.data_seed, device, held_out_offsets=tuple(args.held_out_offsets)
    )

    input_bits = math.ceil(math.log2(CLASSES + 2))
    components = 2 * RANK * 32
    report = {
        "task": "two_phrase_relation_query_with_modular_counterfactual_holdout",
        "condition": "frozen_task_only_and_joint_int4_with_matched_trainable_role_tags",
        "method": "All latent arms train the PBSLM trunk and role/position tags from matching initializations. The frozen arm keeps the autoencoder fixed. The task-only arm updates the encoder using task loss alone. The joint arm updates the encoder and decoder using task loss plus full-phrase reconstruction cross-entropy.",
        "limitations": "Synthetic relation task, frozen random source token embeddings, one predeclared modular-offset split per run (see config.held_out_offsets), and fake int4 quantization with FP32 tensors. The historical 800-step frozen control accidentally disabled tag gradients.",
        "config": {**vars(args), "reconstruction_weight": RECONSTRUCTION_WEIGHT,
                   "frozen_trainable_role_tags": True, "task_only_trainable_encoder": True,
                   "joint_trainable_role_tags": True},
        "torch": torch.__version__,
        "provenance": capture([
            "evaluation/latent_chunk_task_aware_matched.py",
            "evaluation/latent_chunk_task_aware_joint_reconstruction.py",
            "evaluation/latent_chunk_holdout.py",
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
            "components": components,
            "int4_payload_bytes_estimate": components * BITS // 8,
            "dequantized_fp32_input_bytes": components * 4,
            "raw_input_token_count": 2 * PHRASE_LENGTH,
            "input_token_bit_width": input_bits,
            "raw_token_payload_bytes_4bit_packed_estimate": 2 * PHRASE_LENGTH * input_bits // 8,
            "raw_fp32_embedding_input_bytes": 2 * PHRASE_LENGTH * 32 * 4,
            "latent_to_bitpacked_token_payload_ratio": components * BITS / (2 * PHRASE_LENGTH * input_bits),
            "shared_autoencoder_and_decoder_excluded": True,
        },
        "split": {
            "data_seed": args.data_seed,
            "held_out_offsets": sorted(set(args.held_out_offsets)),
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
    for seed in args.model_seeds:
        run = run_seed(seed, train, test, args, device)
        checkpoint_path = checkpoint_dir / f"model_seed_{seed}.json"
        checkpoint = {
            key: report[key]
            for key in ("task", "condition", "config", "split", "code_storage", "provenance")
        }
        checkpoint.update({
            "data_seed": args.data_seed,
            "model_seed": seed,
            "progress": {"complete": True, "checkpoint_model_seed": seed},
            "run": run,
        })
        atomic_write_json(checkpoint_path, checkpoint)
        report["runs"].append(run)
        progress = report["progress"]
        progress["completed_model_seeds"].append(seed)
        progress["checkpoint_files"][str(seed)] = {
            "path": checkpoint_path.relative_to(ROOT).as_posix() if checkpoint_path.is_relative_to(ROOT)
            else str(checkpoint_path),
            "sha256": artifact_sha256(checkpoint_path),
        }
        progress["complete"] = progress["completed_model_seeds"] == progress["requested_model_seeds"]
        atomic_write_json(output_path, report)
        markdown_path = output_path.with_suffix(".md")
        write_markdown(report, markdown_path)
        control_note = (
            "\n## Control Note\n\n"
            "All latent arms train the role and position tags and the PBSLM trunk. "
            "The frozen arm keeps the pretrained encoder fixed. The task-only arm "
            "updates only the encoder with task loss, while the joint arm also updates "
            "the encoder and decoder with phrase-reconstruction loss. "
            "The historical frozen arm accidentally disabled tag gradients, so its "
            "published comparison is not a matched control.\n"
        )
        task_only_lines = [
            "",
            "## Task-Only Encoder Adaptation",
            "",
            "| Model seed | Task-only test | Joint test | Task-only minus joint | Task-only phrase exact | Joint phrase exact |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for record in report["runs"]:
            task_only = record["task_only_adapted_int4_latent"]
            joint = record["joint_int4_latent"]
            task_only_lines.append(
                f"| {record['model_seed']} | {task_only['test']['accuracy']:.4f} | "
                f"{joint['test']['accuracy']:.4f} | "
                f"{task_only['test']['accuracy'] - joint['test']['accuracy']:+.4f} | "
                f"{task_only['int4_test_reconstruction']['both_phrases_exact']:.4f} | "
                f"{joint['int4_test_reconstruction']['both_phrases_exact']:.4f} |"
            )
        atomic_write_text(
            markdown_path,
            markdown_path.read_text(encoding="utf-8")
            + control_note
            + "\n".join(task_only_lines)
            + "\n",
        )
        print(f"model seed {seed}: wrote checkpoint {checkpoint_path}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"wrote aggregate report {output_path}", flush=True)


if __name__ == "__main__":
    main()
