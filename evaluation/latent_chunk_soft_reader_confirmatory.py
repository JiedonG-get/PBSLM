"""Evaluate a soft phrase reader for quantized compact memories.

The frozen phrase autoencoder maps each phrase to an int4-equivalent short
code. Its token distributions are projected into the downstream model's
embedding space, so PBSLM receives continuous embeddings without token IDs.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from evaluation.latent_chunk_ab_thinking import (
    class_logits,
    new_model,
    train_raw,
)
from evaluation.latent_chunk_reconstruction import (
    LatentRoleTags,
    train_autoencoder,
)
from evaluation.latent_chunk_relations import CLASSES, PHRASE_LENGTH, VOCAB, make_data
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]
OFFSET = 4
BITS = 4
RANK = 4
PHRASE_VOCAB = CLASSES + 2


def split_offset_four(seed, device):
    train, standard_test = make_data(seed, device)
    pairs = train["ids"][0::2, PHRASE_LENGTH + 2:PHRASE_LENGTH + 5:2]
    held_out = {(value, (value + OFFSET) % CLASSES) for value in range(CLASSES)}
    selected = torch.tensor(
        [tuple(pair) in held_out for pair in pairs.cpu().tolist()],
        dtype=torch.bool,
        device=device,
    )
    test_rows = selected.repeat_interleave(2)
    held_test = {key: value[test_rows] for key, value in train.items()}
    train = {key: value[~test_rows] for key, value in train.items()}

    train_ids = {tuple(row) for row in train["ids"].cpu().tolist()}
    held_ids = {tuple(row) for row in held_test["ids"].cpu().tolist()}
    assert train_ids.isdisjoint(held_ids)
    assert standard_test["target"].numel() == 3072
    assert set(train["target"].cpu().tolist()) == set(range(CLASSES))
    assert all(tuple(pair) in held_out for pair in pairs[selected].cpu().tolist())
    assert held_test["target"].numel() > 0
    return train, held_test, sorted(held_out)


def quantize_int4(latent):
    return (latent * 7).round().clamp(-7, 7) / 7


@torch.no_grad()
def read_soft_embeddings(model, source_model, encoder, ids, temperature=1.0):
    a, b = encoder.encode(source_model, ids)
    a = quantize_int4(a)
    b = quantize_int4(b)
    distributions = torch.cat((encoder.decode(a), encoder.decode(b)), dim=1)
    probabilities = (distributions / temperature).softmax(-1)
    token_embeddings = F.normalize(model.tok.weight[:PHRASE_VOCAB], dim=-1)
    return probabilities @ token_embeddings


@torch.no_grad()
def evaluate_soft_reader(model, source_model, encoder, data, batch_size=256):
    model.eval()
    source_model.eval()
    encoder.eval()
    predictions = []
    for start in range(0, data["target"].numel(), batch_size):
        sl = slice(start, min(start + batch_size, data["target"].numel()))
        embeds = read_soft_embeddings(model, source_model, encoder, data["ids"][sl])
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


def train_soft_reader(seed, source_model, encoder, train, test, args, device, initial_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_state)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    torch.manual_seed(seed * 1009 + 7)
    rng = torch.Generator(device=device).manual_seed(seed * 997 + 19)
    model.train()

    for _ in range(args.steps):
        indices = torch.randint(
            train["target"].numel(), (args.batch_size,), generator=rng, device=device
        )
        ids = train["ids"][indices]
        with torch.no_grad():
            a, b = encoder.encode(source_model, ids)
            a, b = quantize_int4(a), quantize_int4(b)
            probabilities = torch.cat((encoder.decode(a), encoder.decode(b)), dim=1).softmax(-1)
        token_embeddings = F.normalize(model.tok.weight[:PHRASE_VOCAB], dim=-1)
        embeds = probabilities @ token_embeddings
        output = model(None, 1, inputs_embeds=embeds, sample_noise=False)
        objective = F.cross_entropy(class_logits(model, output), train["target"][indices])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    return {
        "test": evaluate_soft_reader(model, source_model, encoder, test),
        "train": evaluate_soft_reader(model, source_model, encoder, train),
        "model_trainable_parameters_in_optimizer": sum(
            parameter.numel() for parameter in trainable
        ),
        "additional_reader_parameters": 0,
        "embedding_table_parameters": model.tok.weight.numel(),
        "embedding_rows_used_by_reader": PHRASE_VOCAB,
        "embedding_parameters_used_by_reader": PHRASE_VOCAB * model.d,
    }


@torch.no_grad()
def evaluate_direct_int4(model, source_model, encoder, tags, data, batch_size=256):
    model.eval()
    source_model.eval()
    encoder.eval()
    tags.eval()
    predictions = []
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
    return {
        "accuracy": float((predictions == data["target"]).float().mean()),
        "counterfactual_pair_exact": float(pair_exact),
        "examples": data["target"].numel(),
    }


def train_direct_int4(seed, source_model, encoder, train, test, args, device, initial_state):
    model = new_model(seed, device)
    model.load_state_dict(initial_state)
    tags = LatentRoleTags(RANK, model.d).to(device)
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
        "trainable_model_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "role_tag_parameters": sum(parameter.numel() for parameter in tags.parameters()),
    }


@torch.no_grad()
def decode_int4(source_model, encoder, data, batch_size=256):
    decoded = []
    for start in range(0, data["target"].numel(), batch_size):
        ids = data["ids"][start:start + batch_size]
        a, b = encoder.encode(source_model, ids)
        a, b = quantize_int4(a), quantize_int4(b)
        x_ids = encoder.decode(a).argmax(-1)
        y_ids = encoder.decode(b).argmax(-1)
        decoded.append(torch.cat((x_ids, y_ids), dim=1))
    return {"ids": torch.cat(decoded), "target": data["target"], "source": data["source"]}


def token_reconstruction_metrics(predicted_ids, truth_ids):
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
        "y_edge_symbol_accuracy": float(
            (y_prediction[:, 1:5] == y_truth[:, 1:5]).float().mean()
        ),
        "examples": truth_ids.shape[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="experiments/latent_chunk/soft_reader_int4_offset4_3seed_rank4_800.json",
    )
    cli = parser.parse_args()
    args = SimpleNamespace(steps=cli.steps, batch_size=cli.batch_size, lr=cli.lr)
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    report = {
        "task": "two_phrase_key_value_relation_query_with_counterfactual_pairs",
        "condition": "int4_short_codes_soft_decoded_to_continuous_token_embeddings",
        "method": (
            "Reserve offset-4 ordered value pairs before training the phrase autoencoder. Quantize "
            "each normalized rank-4 code with fixed signed 4-bit levels, project the frozen "
            "decoder's per-position class probabilities into the downstream model's trainable "
            "token embedding table, and train the trunk with query loss only. Compare against "
            "direct rank-4 latent inputs, argmax token decoding, and raw-token inputs on the same split."
        ),
        "split": {
            "held_out_pairs": sorted((value, (value + OFFSET) % CLASSES) for value in range(CLASSES)),
            "source": "offset-4 pair rows removed from make_data training rows before encoder or task training",
            "standard_test_split_used": False,
        },
        "code_payload": {
            "bits_per_latent_component": BITS,
            "rank_per_phrase": RANK,
            "phrase_count": 2,
            "components": 2 * RANK * 32,
            "bytes_bit_packed_estimate": 2 * RANK * 32 * BITS // 8,
            "dequantized_fp32_runtime_bytes": 2 * RANK * 32 * 4,
            "scale_metadata_bytes": 0,
            "shared_autoencoder_and_decoder_excluded": True,
            "packed_kernel_or_runtime_memory_saving_tested": False,
        },
        "reader_parameter_accounting": {
            "additional_reader_parameters": 0,
            "reused_shared_decoder_parameters": "counted inside autoencoder_parameters and shared_decoder_parameters per run",
            "model_token_embedding_table_parameters": VOCAB * 32,
            "model_token_embedding_table_shape": [VOCAB, 32],
            "reader_embedding_rows_with_gradients": PHRASE_VOCAB,
            "reader_embedding_parameters_with_gradients": PHRASE_VOCAB * 32,
        },
        "quantization": {
            "method": "round(z * 7).clamp(-7, 7) / 7",
            "in_memory_representation": "dequantized float32 levels",
            "interpretation": "int4-equivalent fake-quantized simulation; no bit-packed runtime path",
        },
        "limitations": (
            "Synthetic task and frozen random source embeddings. The decoder is shared side information, "
            "and the soft reader expands each short code to eight continuous vectors at runtime. The "
            "128-byte figure is a bit-packed payload estimate; this run uses dequantized FP32 values "
            "and does not measure packed-kernel latency or runtime memory. Thus "
            "this tests compressed storage plus continuous readout, not a constant-size model input or "
            "natural-language memory. Only one predeclared offset-4 split and three seeds are evaluated."
        ),
        "config": vars(cli),
        "torch": torch.__version__,
        "provenance": capture([
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
        source_model, encoder, reconstruction = train_autoencoder(
            seed, RANK, train, test, args, device, initial_state
        )
        direct = train_direct_int4(
            seed, source_model, encoder, train, test, args, device, initial_state
        )
        soft_reader = train_soft_reader(
            seed, source_model, encoder, train, test, args, device, initial_state
        )
        decoded_train = decode_int4(source_model, encoder, train)
        decoded_test = decode_int4(source_model, encoder, test)
        decoded = train_raw(seed, decoded_train, decoded_test, args, device, initial_state)
        report["runs"].append({
            "seed": seed,
            "rank": RANK,
            "held_out_value_pairs": held_out_pairs,
            "train_examples": train["target"].numel(),
            "test_examples": test["target"].numel(),
            "latent_payload_bytes_int4": 2 * RANK * 32 * BITS // 8,
            "autoencoder_parameters": sum(p.numel() for p in encoder.parameters()),
            "shared_encoder_parameters": sum(p.numel() for p in encoder.encoder.parameters()),
            "shared_decoder_parameters": sum(p.numel() for p in encoder.decoder.parameters()),
            "reconstruction": reconstruction,
            "int4_quantized_test_reconstruction": token_reconstruction_metrics(
                decoded_test["ids"], test["ids"]
            ),
            "direct_int4_latent_thinking": direct,
            "soft_reader_int4_thinking": soft_reader,
            "argmax_decoded_int4_thinking": decoded,
            "raw_token_baseline": raw_result,
        })
        print(json.dumps(report["runs"][-1], ensure_ascii=False), flush=True)
        del source_model, encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path = ROOT / cli.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
