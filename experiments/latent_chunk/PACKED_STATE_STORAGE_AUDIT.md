# Packed Latent and State Storage Audit

## Scope

This is a byte-accounting audit of the existing synthetic latent-memory experiments. It does not measure packed-kernel speed, GPU memory use, or semantic retrieval quality. The task-aware experiment fake-quantizes latents and continues to hold them as FP32 tensors. The state experiment stores its delta matrix in FP32 and compares it with FP32 key-plus-payload rows. See [`latent_chunk_state.py`](../../evaluation/latent_chunk_state.py) and [`task_aware_joint_recon_int4_offset4_data0_model0-4.json`](task_aware_joint_recon_int4_offset4_data0_model0-4.json).

## Per-Example Payloads

The measured `a/b` representation has two phrases, rank 4 per phrase, and width 32. This is 256 scalar components. The assumed signed int4 representation uses the existing fixed quantization scale `1/7`, codes in `[-7, 7]`, two codes per byte, and no scale metadata.

| Representation | Calculation | Bytes per example |
|---|---:|---:|
| `a/b` codes, ideal packed int4 | `2 x 4 x 32 x 4 / 8` | 128 |
| `a/b` in the current FP32 runtime path | `2 x 4 x 32 x 4` | 1,024 |
| Original 16 synthetic IDs, ideal 4-bit packing | `16 x ceil(log2(10)) / 8` | 8 |
| Original IDs in current `torch.long` storage | `16 x 8` | 128 |
| Sixteen width-32 FP32 token embeddings | `16 x 32 x 4` | 2,048 |

The 10-symbol alphabet makes 4 bits per token sufficient for this synthetic task. Its 128-byte estimate for `a/b` latents is 16 times the 8-byte packed-ID estimate. The latent code is smaller than the FP32 embedding input, but the experiment neither stores the codes in packed form nor measures an end-to-end memory reduction.

The shared phrase autoencoder has 38,224 parameters: 24,768 encoder and 13,456 decoder parameters. At FP32 this is 152,896 bytes, about 149.3 KiB, excluding framework and allocator overhead. This shared cost is excluded from per-example payload estimates. For the proposed four slots `a/b/c/d` at rank 4 and width 32, the projected payload is 512 components, or 256 ideal int4 bytes and 2,048 FP32 bytes. This four-slot figure is arithmetic only. The task-aware joint experiment writes and reads `a/b`, not a persisted `a/b/c/d` code.

## State Versus Table Accounting

The existing delta-state evaluator uses 16-dimensional FP32 keys, or 64 bytes per key, and a dense FP32 state with `16 x payload_width` elements. The following comparison assumes the table stores each key plus an ideally packed code, while the state remains FP32. It excludes metadata, alignment, allocator overhead, shared encoder/decoder weights, and any key index.

| Payload | State width | FP32 state | Packed table row, key + code | Break-even episode length |
|---|---:|---:|---:|---:|
| `a/b`, rank 4 | 256 | 16,384 B | 192 B | 86 items |
| `a/b/c/d`, rank 4 | 512 | 32,768 B | 320 B | 103 items |

Break-even is the first integer length where the table bytes equal or exceed the fixed state bytes. At 96 items, the `a/b` table is 18,432 B versus a 16,384 B state. The four-slot table is 30,720 B versus a 32,768 B state, so the dense state is larger at that length. These are storage projections, not measured retrieval equivalence. The table retains exact codes per item; the delta state is lossy and can interfere across writes.

As an existing reference, the rank-1 four-slot random-key screen reported an 8,192-byte FP32 state versus 55,296 bytes for 96 FP32 key-plus-payload rows. Its oldest-item payload cosine was 0.074 and state-query accuracy was near chance. That result tests IID random keys and recency interference. It does not show learned semantic addressing or the behavior of packed rank-4 codes.

## Packing Prototype

A one-off pure-Python prototype tested signed nibble packing on 256 seeded values in `[-7, 7]`. It placed the even-index value in the low nibble and odd-index value in the high nibble, then sign-extended on decode. The result was 128 bytes and an exact integer round-trip. This verifies the byte arithmetic only. It did not exercise the repository's PyTorch path, actual encoder outputs, decoder outputs, state writes, CUDA, or latency.

```python
import random

values = [random.Random(20261119 + index).randint(-7, 7) for index in range(256)]
packed = bytearray((len(values) + 1) // 2)
for index, value in enumerate(values):
    packed[index // 2] |= (value & 0x0F) << (4 if index % 2 else 0)

decoded = []
for index in range(len(values)):
    nibble = (packed[index // 2] >> (4 if index % 2 else 0)) & 0x0F
    decoded.append(nibble - 16 if nibble >= 8 else nibble)

assert decoded == values
assert len(packed) == 128
```

The prototype deliberately makes no runtime-speed claim. A deployed implementation would still need packed tensor serialization, unpacking or a compatible low-bit kernel, validation on actual latent values, and retrieval tests under the same byte budget.
