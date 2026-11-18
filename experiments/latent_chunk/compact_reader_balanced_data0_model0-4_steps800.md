# Compact Reader Evaluation

- Status: true (10 completed runs)
- Model seeds: [0, 1, 2, 3, 4]
- Training steps per arm: 800
- Results are mean ± sample SD across model seeds on each fixed balanced split.

| Held-out offsets | Input condition | Vectors | Test accuracy | Pair exact |
|---|---|---:|---:|---:|
| [3, 5] | Raw tokens | 16 | 78.44 ± 14.22% | 60.63 ± 25.14% |
| [3, 5] | Direct int4 a/b | 8 | 30.83 ± 0.97% | 6.50 ± 0.84% |
| [3, 5] | Compact soft reader | 8 | 97.09 ± 0.40% | 94.52 ± 0.57% |
| [3, 5] | Full soft reader | 16 | 74.58 ± 15.43% | 54.36 ± 26.51% |
| [4] | Raw tokens | 16 | 85.22 ± 14.27% | 72.79 ± 24.76% |
| [4] | Direct int4 a/b | 8 | 29.15 ± 1.58% | 5.81 ± 1.39% |
| [4] | Compact soft reader | 8 | 96.07 ± 2.22% | 92.63 ± 3.99% |
| [4] | Full soft reader | 16 | 80.34 ± 12.75% | 64.36 ± 21.04% |

The rank-4 a/b code has 128 B of ideal int4 payload, while its dequantized FP32 training representation uses 1,024 B. The compact reader expands to eight 32-dimensional FP32 vectors (1,024 B) and the full reader to sixteen (2,048 B). These figures exclude shared encoder/decoder and reader parameters. The synthetic 16-token input can be represented in 8 B with ideal 4-bit packing, so the latent payload is not smaller than packed source IDs.

The compact reader merges adjacent decoded token pairs with one shared linear layer, then adds trainable phrase-role and slot tags. It is trained with task loss only. This synthetic benchmark does not establish natural-language retrieval or actual packed-runtime memory and latency savings.
