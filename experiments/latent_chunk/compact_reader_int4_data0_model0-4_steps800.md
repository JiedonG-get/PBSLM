# Packed Int4 Compact Reader

- Status: true (10 completed runs)
- Model seeds: [0, 1, 2, 3, 4]
- Task steps per arm: 800
- Results are mean ± sample SD across model seeds on each fixed balanced split.

| Held-out offsets | Input | Test accuracy | Counterfactual pair exact |
|---|---|---:|---:|
| [3, 5] | Raw tokens | 78.44 ± 14.22% | 60.63 ± 25.14% |
| [3, 5] | Compact FP32 vectors | 97.09 ± 0.40% | 94.52 ± 0.57% |
| [3, 5] | FP32-trained vectors, packed int4 at test | 96.17 ± 0.60% | 92.77 ± 0.98% |
| [3, 5] | Quantization-aware trained vectors | 97.17 ± 0.56% | 94.64 ± 0.93% |
| [4] | Raw tokens | 85.22 ± 14.27% | 72.79 ± 24.76% |
| [4] | Compact FP32 vectors | 96.07 ± 2.22% | 92.63 ± 3.99% |
| [4] | FP32-trained vectors, packed int4 at test | 95.18 ± 2.40% | 90.90 ± 4.32% |
| [4] | Quantization-aware trained vectors | 95.76 ± 2.34% | 92.01 ± 4.31% |

The post-hoc packed row uses the same checkpoint as the FP32 row, isolating int4 quantization at inference. The quantization-aware row trains with fake-int4 quantization and a straight-through gradient. Evaluation packs signed 4-bit values into bytes and unpacks those bytes before the trunk. Each example stores eight 32-dimensional vectors in 128 bytes, excluding shared encoder, merger, role tags, and trunk parameters.

The compact representation is produced from the decoded phrase by a learned pair merger and role tags. The experiment tests quantized task-specific memory vectors in this synthetic task. The source phrase decoder is still used when writing the representation. Ideal fixed-width packing of the 16 source IDs requires 8 bytes, so this is not a storage win over source text. It does not measure a production runtime cache or latency.
