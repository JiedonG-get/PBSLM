"""Test fixed-size episodic state for compact PBSLM phrase vectors.

Run from the repository root:
    python -m evaluation.latent_chunk_compact_state --device cuda

Small CPU run:
    python -m evaluation.latent_chunk_compact_state --device cpu --seeds 0
      --held-out-offsets 4 --pretrain-steps 1 --teacher-steps 1 --steps 1
      --batch-size 4 --episode-batch-size 2 --train-lengths 3 --test-lengths 3 5
      --test-episodes 2 --output <temporary-path.json>
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from evaluation.latent_chunk_ab_thinking import class_logits, new_model
from evaluation.latent_chunk_compact_int4 import make_offsets_sets, pack_int4, unpack_int4
from evaluation.latent_chunk_compact_student import (
    score_predictions,
    teacher_features,
    train_teacher,
    write_json_atomic,
)
from evaluation.latent_chunk_holdout import make_held_out_data
from evaluation.latent_chunk_reconstruction import train_autoencoder
from evaluation.latent_chunk_relations import CLASSES
from evaluation.latent_chunk_soft_reader_confirmatory import RANK
from evaluation.latent_chunk_state import KEY_DIM, WRITE_STRENGTH
from evaluation.provenance import capture


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "evaluation/latent_chunk_compact_state.py",
    "evaluation/latent_chunk_compact_student.py",
    "evaluation/latent_chunk_compact_int4.py",
    "evaluation/latent_chunk_compact_reader.py",
    "evaluation/latent_chunk_soft_reader_confirmatory.py",
    "evaluation/latent_chunk_reconstruction.py",
    "evaluation/latent_chunk_ab_thinking.py",
    "evaluation/latent_chunk_holdout.py",
    "evaluation/latent_chunk_relations.py",
    "evaluation/latent_chunk_state.py",
    "evaluation/provenance.py",
    "models/model_v2.py",
)


def write_state(keys, values, strength):
    # Functional form of the existing delta update, with one matrix per episode.
    state = values.new_zeros(values.shape[0], keys.shape[-1], values.shape[-1])
    for key, value in zip(keys.unbind(1), values.unbind(1)):
        residual = value - torch.bmm(key.unsqueeze(1), state).squeeze(1)
        state = state + strength * key.unsqueeze(-1) * residual.unsqueeze(1)
    return state


class LearnedStateRead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.write_logit = nn.Parameter(torch.tensor(math.log(WRITE_STRENGTH / (1 - WRITE_STRENGTH))))
        self.read_projection = nn.Linear(width, width)
        nn.init.zeros_(self.read_projection.weight)
        nn.init.zeros_(self.read_projection.bias)

    def read(self, query, state):
        value = torch.bmm(query, state)
        return value + self.read_projection(value)


@torch.no_grad()
def prepare_split(teacher, source_model, encoder, split):
    features = torch.cat([
        teacher_features(teacher, source_model, encoder, ids)[1]
        for ids in split["ids"].split(256)
    ])
    a = features[:, 0].reshape(-1, 2, RANK, features.shape[-1])
    b_pair = features[:, 1].reshape(-1, 2, RANK, features.shape[-1])
    torch.testing.assert_close(b_pair[:, 0], b_pair[:, 1], rtol=1e-5, atol=1e-7)
    b = b_pair[:, 0].flatten(1)
    packed_b = pack_int4(b)
    return {
        "a": a,
        "b": b,
        "packed_b": packed_b,
        "dequantized_b": unpack_int4(packed_b, b.shape),
        "target": split["target"].view(-1, 2),
    }


def sample_episode(split, batch, length, generator):
    pool = split["b"].shape[0]
    indices = torch.stack([
        torch.randperm(pool, generator=generator)[:length] for _ in range(batch)
    ])
    keys = F.normalize(torch.randn(batch, length, KEY_DIM, generator=generator), dim=-1)
    device = split["b"].device
    return indices.to(device), keys.to(device)


def train_state(split, args, generator):
    width = split["b"].shape[-1]
    learner = LearnedStateRead(width).to(split["b"].device)
    optimizer = torch.optim.AdamW(learner.parameters(), lr=args.state_lr, weight_decay=1e-4)
    for step in range(args.steps):
        length = args.train_lengths[step % len(args.train_lengths)]
        indices, keys = sample_episode(split, args.episode_batch_size, length, generator)
        values = split["dequantized_b"][indices]
        state = write_state(keys, values, learner.write_logit.sigmoid())
        positions = torch.randint(length, (indices.shape[0], 1), generator=generator).to(keys.device)
        rows = torch.arange(indices.shape[0], device=keys.device).unsqueeze(1)
        retrieved = learner.read(keys[rows, positions], state)
        target = values[rows, positions]
        retrieved_vectors = retrieved.reshape(-1, RANK, width // RANK)
        target_vectors = target.reshape_as(retrieved_vectors)
        loss = (retrieved - target).square().sum(-1).mean() / target.square().sum(-1).mean().clamp_min(1e-8)
        loss = loss + (1 - F.cosine_similarity(retrieved_vectors, target_vectors, dim=-1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    learner.eval()
    return learner, float(loss.detach())


def evaluate_payload(teacher, a, b, target, packed_reference):
    # Both x branches receive the same recalled y payload.
    branches = a.reshape(-1, 2, RANK, a.shape[-1]).flatten(0, 1)
    memory = b.reshape(-1, RANK, a.shape[-1]).repeat_interleave(2, dim=0)
    embeds = teacher["tags"].apply(branches, memory)
    model = teacher["model"]
    predictions = torch.cat([
        class_logits(model, model(None, 1, inputs_embeds=batch, sample_noise=False)).argmax(-1)
        for batch in embeds.split(256)
    ])
    metrics = score_predictions(predictions, target.flatten())
    metrics.update({
        "payload_mse_to_packed_table": float(F.mse_loss(b, packed_reference)),
        "payload_cosine_to_packed_table": float(F.cosine_similarity(b, packed_reference, dim=-1).mean()),
    })
    return metrics


@torch.no_grad()
def evaluate_state(teacher, learner, split, args, episode_seed):
    results = {}
    for length in args.test_lengths:
        generator = torch.Generator(device="cpu").manual_seed(episode_seed + length)
        indices, keys = sample_episode(split, args.test_episodes, length, generator)
        values = split["dequantized_b"][indices]
        fixed = write_state(keys, values, WRITE_STRENGTH)
        learned = write_state(keys, values, learner.write_logit.sigmoid())
        age_results = {}
        for label, position in (("oldest", 0), ("middle", (length - 1) // 2), ("latest", length - 1)):
            query = keys[:, position:position + 1]
            wrong_position = (position + length // 2) % length
            wrong_query = keys[:, wrong_position:wrong_position + 1]
            target_indices = indices[:, position]
            table_packed = values[:, position]
            arms = {
                "pristine_fp32": split["b"][target_indices],
                "exact_packed_table": table_packed,
                "fixed_delta": torch.bmm(query, fixed).squeeze(1),
                "learned_delta": learner.read(query, learned).squeeze(1),
                "learned_reset": learner.read(query, torch.zeros_like(learned)).squeeze(1),
                "learned_wrong_address": learner.read(wrong_query, learned).squeeze(1),
            }
            age_results[label] = {
                "age_writes": length - 1 - position,
                "conditions": {
                    arm: evaluate_payload(
                        teacher, split["a"][target_indices], b,
                        split["target"][target_indices], table_packed,
                    )
                    for arm, b in arms.items()
                },
            }
        key_bytes = keys.shape[-1] * keys.element_size()
        packed_b_bytes = split["packed_b"].shape[-1] * split["packed_b"].element_size()
        results[str(length)] = {
            "episodes": args.test_episodes,
            "state_bytes_per_episode": fixed[0].numel() * fixed.element_size(),
            "packed_b_bytes_per_table_entry": packed_b_bytes,
            "address_bytes_per_table_entry": key_bytes,
            "exact_table_bytes_per_episode": length * (key_bytes + packed_b_bytes),
            "query_ages": age_results,
        }
    return results


def run_split(args, offsets, seed, train, test, held_out_pairs, device):
    initial_state = copy.deepcopy(new_model(seed, device).state_dict())
    pretrain_args = argparse.Namespace(
        steps=args.pretrain_steps, batch_size=args.batch_size, lr=args.lr,
    )
    source_model, encoder, reconstruction = train_autoencoder(
        seed, RANK, train, test, pretrain_args, device, initial_state,
    )
    teacher = train_teacher(seed, source_model, encoder, train, args, device, initial_state)
    train_features = prepare_split(teacher, source_model, encoder, train)
    test_features = prepare_split(teacher, source_model, encoder, test)
    train_seed = 1_000_003 * seed + 200_003
    test_seed = 1_000_003 * seed + 500_009
    generator = torch.Generator(device="cpu").manual_seed(train_seed)
    learner, fit_loss = train_state(train_features, args, generator)
    record = {
        "held_out_offsets": offsets,
        "held_out_pairs": held_out_pairs,
        "seed": seed,
        "train_examples": train["target"].numel(),
        "test_examples": test["target"].numel(),
        "reconstruction_pretraining": reconstruction,
        "episode_train_seed": train_seed,
        "episode_test_seed_base": test_seed,
        "learned_write_strength": float(learner.write_logit.sigmoid().detach()),
        "learned_state_parameters": sum(p.numel() for p in learner.parameters()),
        "last_train_loss": fit_loss,
        "evaluation": evaluate_state(teacher, learner, test_features, args, test_seed),
    }
    if args.save_checkpoints:
        output_path = ROOT / args.output
        directory = output_path.with_name(f"{output_path.stem}_checkpoints")
        directory.mkdir(parents=True, exist_ok=True)
        split_label = "-".join(map(str, offsets))
        path = directory / f"offsets{split_label}_seed{seed}.pt"
        temporary_path = path.with_suffix(".pt.tmp")
        torch.save({
            "format": "pbslm.compact_state.research.v1",
            "held_out_offsets": offsets,
            "seed": seed,
            "source_model": source_model.state_dict(),
            "teacher": {name: module.state_dict() for name, module in teacher.items()},
            "learner": learner.state_dict(),
        }, temporary_path)
        temporary_path.replace(path)
        record["checkpoint"] = {
            "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--held-out-offsets", nargs="+", type=int, default=[4, 3, 5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--pretrain-steps", type=int, default=800)
    parser.add_argument("--teacher-steps", type=int, default=800)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episode-batch-size", type=int, default=16)
    parser.add_argument("--test-episodes", type=int, default=64)
    parser.add_argument("--train-lengths", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--test-lengths", nargs="+", type=int, default=[8, 32, 96])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--state-lr", type=float, default=0.002)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="experiments/latent_chunk/compact_reader_state.json")
    parser.add_argument("--save-checkpoints", action="store_true")
    cli = parser.parse_args()
    if min(cli.pretrain_steps, cli.teacher_steps, cli.steps, cli.batch_size,
           cli.episode_batch_size, cli.test_episodes) <= 0:
        parser.error("steps, batches, and episode count must be positive")
    for offsets in cli.held_out_offsets:
        if offsets not in range(1, CLASSES):
            parser.error(f"held-out offsets must be in 1..{CLASSES - 1}")
    if len(set(cli.seeds)) != len(cli.seeds) or len(set(cli.held_out_offsets)) != len(cli.held_out_offsets):
        parser.error("seeds and offsets must be unique")

    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    output = ROOT / cli.output
    if output.exists():
        parser.error(f"output already exists: {output}")
    offsets_sets = make_offsets_sets(cli.held_out_offsets)
    report = {
        "task": "fixed_state_retrieval_of_compact_y_for_pbsml_relation_reasoning",
        "config": vars(cli),
        "provenance": capture(SOURCES),
        "fixed_write_strength": WRITE_STRENGTH,
        "key_width": KEY_DIM,
        "independent_uniform_guess_reference": {
            "accuracy": 1 / CLASSES,
            "counterfactual_pair_exact": 1 / CLASSES ** 2,
        },
        "method": (
            "Write only the packed compact b(y) vector group under a fresh random FP32 address. "
            "Read b from fixed or learned FP32 delta state and combine it with the live a(x) branch. "
            "The train-only state loss reconstructs b, and each comparison shares a frozen compact "
            "teacher and the same held-out episodes. The exact table holds quantized b and the address."
        ),
        "limitations": (
            "Fixed supplied addresses and a provided write segmentation. The existing vocabulary "
            "decoder generates compact features during encoding. The state reads have no token access. "
            "The exact table uses evaluator lookup as a storage reference. Counts include persistent "
            "state or table tensor payloads, excluding model parameters, metadata, and activations. "
            "Reset and wrong-address rows are empirical intervention controls. No automatic salience, "
            "semantic addressing, online parameter learning, or real runtime memory is evaluated."
        ),
        "progress": {"complete": False},
        "runs": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for offsets in offsets_sets:
        train, test, held_out_pairs = make_held_out_data(cli.data_seed, device, held_out_offsets=offsets)
        for lengths, split in ((cli.train_lengths, train), (cli.test_lengths, test)):
            if len(set(lengths)) != len(lengths) or min(lengths) < 3 or max(lengths) > split["target"].numel() // 2:
                parser.error("episode lengths must be unique and in 3..number of counterfactual groups")
        for seed in cli.seeds:
            print(f"offsets {offsets}, model seed {seed}: fitting state", flush=True)
            run = run_split(cli, offsets, seed, train, test, held_out_pairs, device)
            report["runs"].append(run)
            write_json_atomic(output, report)
    report["progress"]["complete"] = True
    write_json_atomic(output, report)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
