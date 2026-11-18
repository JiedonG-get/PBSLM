"""Fixed-state retrieval control for four-slot latent chunk payloads.

This evaluation-only control stores flattened payloads in one FP32 delta-rule
state per episode. It uses IID random normalized keys, without learning a key
encoder or readout. Exact-table results use perfect key lookup and the original
payload, so they are an upper-bound storage baseline rather than a learned read.
"""

import math

import torch
import torch.nn.functional as F


EPISODE_LENGTHS = (8, 32, 96)
KEY_DIM = 16
WRITE_STRENGTH = 0.865
EPISODES = 64


def _prediction_metrics(read_payload, pristine_payload, target, predict):
    state_logits = predict(read_payload)
    table_logits = predict(pristine_payload)
    if state_logits.ndim != 2 or state_logits.shape[0] != target.numel():
        raise ValueError("predict must return logits with shape [B, C]")
    if table_logits.shape != state_logits.shape:
        raise ValueError("predict must return matching logits for state and table payloads")

    read_flat = read_payload.float().flatten(1)
    original_flat = pristine_payload.float().flatten(1)
    cosine = F.cosine_similarity(read_flat, original_flat, dim=-1, eps=1e-8)
    return {
        "state_accuracy": float((state_logits.argmax(-1) == target).float().mean().item()),
        "exact_table_accuracy": float((table_logits.argmax(-1) == target).float().mean().item()),
        "state_prediction_matches_exact_table": float(
            (state_logits.argmax(-1) == table_logits.argmax(-1)).float().mean().item()
        ),
        "payload_mse": float((read_flat - original_flat).square().mean(-1).mean().item()),
        "payload_cosine": float(cosine.mean().item()),
    }


@torch.no_grad()
def evaluate_state(payload, targets, predict, seed=0):
    """Evaluate oldest/latest retrieval over 64 episodes per length.

    Args:
        payload: Float tensor shaped [N, 4, r, d]. The four slots are untagged.
        targets: Integer tensor shaped [N].
        predict: Callable mapping [B, 4, r, d] payloads to [B, C] logits.
        seed: Controls random episode sampling and IID random retrieval keys.

    Returns:
        A JSON-serializable dictionary with mean retrieval quality and byte counts.
    """
    if not isinstance(payload, torch.Tensor) or payload.ndim != 4:
        raise ValueError("payload must be a tensor shaped [N, 4, r, d]")
    if payload.shape[1] != 4 or not payload.is_floating_point():
        raise ValueError("payload must have four floating-point slots")
    if not isinstance(targets, torch.Tensor) or targets.shape != (payload.shape[0],):
        raise ValueError("targets must be a rank-one tensor matching payload count")
    if targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("targets must use an integer dtype")
    if payload.shape[2] < 1 or payload.shape[3] < 1:
        raise ValueError("latent dimensions r and d must be positive")

    target_cpu = targets.detach().to(device="cpu", dtype=torch.long)
    if target_cpu.numel() < max(EPISODE_LENGTHS):
        raise ValueError(
            f"each episode requires at least {max(EPISODE_LENGTHS)} distinct samples, "
            f"received {target_cpu.numel()}"
        )

    episode_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    key_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 104729)
    device = payload.device
    item_shape = payload.shape[1:]
    value_width = math.prod(item_shape)
    key_bytes_per_item = KEY_DIM * 4
    state_bytes = value_width * key_bytes_per_item
    payload_bytes_per_item = value_width * payload.element_size()
    results = {}

    for length in EPISODE_LENGTHS:
        # Each row is an independent random episode, with no repeated item
        # inside that episode. Rows may reuse items from the same held-out pool.
        episode_indices = torch.stack([
            torch.randperm(target_cpu.numel(), generator=episode_generator)[:length]
            for _ in range(EPISODES)
        ])
        labels = target_cpu[episode_indices].to(device=device)
        episode_indices = episode_indices.to(device=device)
        values = payload[episode_indices].float().reshape(EPISODES, length, value_width)
        keys = torch.randn(EPISODES, length, KEY_DIM, generator=key_generator, dtype=torch.float32)
        keys = F.normalize(keys, dim=-1).to(device=device, dtype=torch.float32)

        state = torch.zeros(EPISODES, KEY_DIM, value_width, device=device, dtype=torch.float32)
        for step in range(length):
            key = keys[:, step]
            value = values[:, step]
            prediction = torch.bmm(key.unsqueeze(1), state).squeeze(1)
            residual = value - prediction
            state.add_(WRITE_STRENGTH * torch.bmm(key.unsqueeze(2), residual.unsqueeze(1)))

        oldest_read = torch.bmm(keys[:, 0].unsqueeze(1), state).squeeze(1)
        latest_read = torch.bmm(keys[:, -1].unsqueeze(1), state).squeeze(1)
        oldest_read = oldest_read.reshape(EPISODES, *item_shape).to(payload.dtype)
        latest_read = latest_read.reshape(EPISODES, *item_shape).to(payload.dtype)
        oldest_original = payload[episode_indices[:, 0]]
        latest_original = payload[episode_indices[:, -1]]

        flat_table_bytes = length * (key_bytes_per_item + payload_bytes_per_item)
        results[str(length)] = {
            "episode_count": EPISODES,
            "state_bytes_fp32_per_episode": int(state_bytes),
            "exact_table_bytes_keys_plus_payload_per_episode": int(flat_table_bytes),
            "state_to_table_byte_ratio": float(state_bytes / flat_table_bytes),
            "oldest_query_mean": _prediction_metrics(
                oldest_read, oldest_original, labels[:, 0], predict
            ),
            "latest_query_mean": _prediction_metrics(
                latest_read, latest_original, labels[:, -1], predict
            ),
        }

    return {
        "method": "fixed_fp32_delta_state",
        "write_strength": WRITE_STRENGTH,
        "key_distribution": "iid_standard_normal_l2_normalized_16d",
        "state_dtype": "float32",
        "episode_count_per_length": EPISODES,
        "payload_shape_per_item": list(payload.shape[1:]),
        "payload_source_dtype": str(payload.dtype),
        "exact_table_prediction_source": "perfect_key_lookup_of_pristine_payload",
        "sampling": "without_replacement_within_episode_from_held_out_pool",
        "table_predictions_match_pristine_by_construction": True,
        "seed": int(seed),
        "episodes": results,
    }
