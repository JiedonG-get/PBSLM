"""Run rank-1/2 soft-reader controls on the locked offset-4 split."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from evaluation.latent_chunk_ab_thinking import class_logits, new_model, train_raw
from evaluation.latent_chunk_reconstruction import LatentRoleTags, train_autoencoder
from evaluation.latent_chunk_relations import CLASSES, PHRASE_LENGTH, VOCAB
from evaluation.latent_chunk_soft_reader_confirmatory import (
    BITS,
    decode_int4,
    quantize_int4,
    split_offset_four,
    token_reconstruction_metrics,
    train_soft_reader,
)
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]


def evaluate_direct_int4(model, source_model, encoder, tags, data, batch_size=256):
    model.eval()
    source_model.eval()
    encoder.eval()
    tags.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, data["target"].numel(), batch_size):
            sl = slice(start, min(start + batch_size, data["target"].numel()))
            a, b = encoder.encode(source_model, data["ids"][sl])
            embeddings = tags.apply(quantize_int4(a), quantize_int4(b))
            output = model(None, 1, inputs_embeds=embeddings, sample_noise=False)
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


def train_direct_int4(seed, rank, source_model, encoder, train, test, args, device, initial_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_state)
    tags = LatentRoleTags(rank, model.d).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable += list(tags.parameters())
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
        with torch.no_grad():
            a, b = encoder.encode(source_model, ids)
            a, b = quantize_int4(a), quantize_int4(b)
        output = model(None, 1, inputs_embeds=tags.apply(a, b), sample_noise=False)
        objective = F.cross_entropy(class_logits(model, output), train["target"][indices])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    return {
        "test": evaluate_direct_int4(model, source_model, encoder, tags, test),
        "train": evaluate_direct_int4(model, source_model, encoder, tags, train),
        "model_trainable_parameters_in_optimizer": sum(p.numel() for p in trainable),
        "role_tag_parameters": sum(p.numel() for p in tags.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2], choices=[1, 2])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="experiments/latent_chunk/soft_reader_int4_offset4_rank12_3seed_800.json",
    )
    cli = parser.parse_args()
    args = SimpleNamespace(steps=cli.steps, batch_size=cli.batch_size, lr=cli.lr)
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    report = {
        "task": "two_phrase_key_value_relation_query_with_counterfactual_pairs",
        "condition": "int4_soft_reader_rank_sweep_on_offset4_holdout",
        "method": (
            "Use the same locked offset-4 split, fixed-scale int4-equivalent quantization, and "
            "frozen phrase decoder as the rank-4 confirmatory experiment. For each requested rank, "
            "compare direct compact latents, continuous soft-decoded embeddings, argmax-decoded "
            "token IDs, and raw-token training. All task training uses query loss only."
        ),
        "split": {
            "held_out_pairs": sorted(
                (value, (value + 4) % CLASSES) for value in range(CLASSES)
            ),
            "source": "pair rows removed from make_data training rows before encoder or task training",
            "standard_test_split_used": False,
        },
        "quantization": {
            "bits_per_component": BITS,
            "method": "round(z * 7).clamp(-7, 7) / 7",
            "in_memory_representation": "dequantized float32 levels",
            "packed_runtime_path_tested": False,
        },
        "payload_bytes_by_rank": {
            str(rank): 2 * rank * 32 * BITS // 8 for rank in cli.ranks
        },
        "reader_parameter_accounting": {
            "additional_reader_parameters": 0,
            "shared_decoder_parameters_reported_per_run": True,
            "model_token_embedding_table_parameters": VOCAB * 32,
            "reader_embedding_rows_with_gradients": CLASSES + 2,
            "reader_embedding_parameters_with_gradients": (CLASSES + 2) * 32,
        },
        "limitations": (
            "Synthetic task and frozen random source embeddings. The shared autoencoder decoder "
            "is excluded from per-example payload. Reported bit-packed bytes are estimates, while "
            "computation uses dequantized FP32 tensors. The soft reader expands each phrase code "
            "to eight continuous vectors and does not establish natural-language memory utility."
        ),
        "config": vars(cli),
        "torch": torch.__version__,
        "provenance": capture([
            "evaluation/latent_chunk_soft_reader_rank_sweep.py",
            "evaluation/latent_chunk_soft_reader_confirmatory.py",
            "evaluation/latent_chunk_reconstruction.py",
            "evaluation/latent_chunk_ab_thinking.py",
            "evaluation/latent_chunk_relations.py",
            "evaluation/provenance.py",
            "models/model.py",
            "models/model_v2.py",
        ]),
        "runs": [],
    }

    for seed in cli.seeds:
        train, test, held_out_pairs = split_offset_four(seed, device)
        initial_model = new_model(seed, device)
        initial_state = copy.deepcopy(initial_model.state_dict())
        del initial_model
        raw_result = train_raw(seed, train, test, args, device, initial_state)

        for rank in cli.ranks:
            source_model, encoder, reconstruction = train_autoencoder(
                seed, rank, train, test, args, device, initial_state
            )
            direct = train_direct_int4(
                seed, rank, source_model, encoder, train, test, args, device, initial_state
            )
            soft_reader = train_soft_reader(
                seed, source_model, encoder, train, test, args, device, initial_state
            )
            decoded_train = decode_int4(source_model, encoder, train)
            decoded_test = decode_int4(source_model, encoder, test)
            decoded = train_raw(seed, decoded_train, decoded_test, args, device, initial_state)
            record = {
                "seed": seed,
                "rank": rank,
                "held_out_value_pairs": held_out_pairs,
                "train_examples": train["target"].numel(),
                "test_examples": test["target"].numel(),
                "latent_payload_bytes_int4_estimate": 2 * rank * 32 * BITS // 8,
                "latent_payload_bytes_dequantized_fp32": 2 * rank * 32 * 4,
                "autoencoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
                "shared_encoder_parameters": sum(parameter.numel() for parameter in encoder.encoder.parameters()),
                "shared_decoder_parameters": sum(parameter.numel() for parameter in encoder.decoder.parameters()),
                "reconstruction": reconstruction,
                "int4_quantized_test_reconstruction": token_reconstruction_metrics(
                    decoded_test["ids"], test["ids"]
                ),
                "direct_int4_latent_thinking": direct,
                "soft_reader_int4_thinking": soft_reader,
                "argmax_decoded_int4_thinking": decoded,
                "raw_token_baseline": raw_result,
            }
            report["runs"].append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            del source_model, encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()

    output_path = ROOT / cli.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
