# Task-Aware Joint Reconstruction Experiment

- Status: `complete` (5/5 model seeds completed)
- Data seed: `0`
- Model seeds requested: `[0, 1, 2, 3, 4]`
- Split: offset-4 ordered value pairs held out before training (3384 train, 712 test rows)
- Rank: `4` per phrase, with `8` input vectors total
- Steps: `800` reconstruction pretraining, `800` frozen-latent trunk training, `800` joint tuning, `800` raw baseline
- Objective: task cross-entropy + `1.0` x full-phrase reconstruction cross-entropy
- Quantization: signed int4-equivalent fake quantization, with FP32 tensors and straight-through gradients

## Task Results

| Model seed | Raw train | Raw test | Frozen train | Frozen test | Joint train | Joint test | Raw pair-exact | Frozen pair-exact | Joint pair-exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9515 | 0.8610 | 0.9944 | 0.2528 | 0.9498 | 0.3778 | 0.7416 | 0.0393 | 0.0843 |
| 1 | 0.8679 | 0.7317 | 0.9953 | 0.2753 | 0.9474 | 0.3567 | 0.5112 | 0.0618 | 0.0983 |
| 2 | 0.9642 | 0.9157 | 0.9965 | 0.2219 | 0.9716 | 0.4073 | 0.8399 | 0.0337 | 0.1236 |
| 3 | 0.8614 | 0.7079 | 0.9956 | 0.2008 | 0.9574 | 0.3371 | 0.5028 | 0.0421 | 0.1067 |
| 4 | 0.8023 | 0.6348 | 0.9973 | 0.2374 | 0.9524 | 0.3343 | 0.3315 | 0.0562 | 0.0871 |

## Int4 Reconstruction

| Model seed | Before task tuning, token acc. | Before, both phrases exact | After joint tuning, token acc. | After, both phrases exact |
|---:|---:|---:|---:|---:|
| 0 | 0.9971 | 0.9579 | 0.9970 | 0.9551 |
| 1 | 0.9934 | 0.9059 | 0.9953 | 0.9242 |
| 2 | 0.9918 | 0.8933 | 0.9933 | 0.9045 |
| 3 | 0.9915 | 0.8806 | 0.9953 | 0.9340 |
| 4 | 0.9954 | 0.9340 | 0.9954 | 0.9298 |

## Held-Out Pair Accuracy, Model Seed 0

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.9000 | 0.2875 | 0.4875 |
| 1->5 | 0.7647 | 0.2843 | 0.4412 |
| 2->6 | 0.8091 | 0.3000 | 0.4636 |
| 3->7 | 0.8981 | 0.3333 | 0.3056 |
| 4->0 | 0.9688 | 0.1250 | 0.3281 |
| 5->1 | 0.8000 | 0.1875 | 0.3125 |
| 6->2 | 0.8718 | 0.1923 | 0.2949 |
| 7->3 | 0.9222 | 0.2333 | 0.3556 |

## Held-Out Pair Accuracy, Model Seed 1

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.7250 | 0.2375 | 0.3625 |
| 1->5 | 0.7255 | 0.3529 | 0.4216 |
| 2->6 | 0.6273 | 0.3273 | 0.3182 |
| 3->7 | 0.8241 | 0.2778 | 0.4352 |
| 4->0 | 0.6562 | 0.2500 | 0.3594 |
| 5->1 | 0.7625 | 0.2000 | 0.4000 |
| 6->2 | 0.7179 | 0.3205 | 0.2949 |
| 7->3 | 0.8000 | 0.2000 | 0.2444 |

## Held-Out Pair Accuracy, Model Seed 2

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.9000 | 0.2125 | 0.4250 |
| 1->5 | 0.8725 | 0.1667 | 0.4510 |
| 2->6 | 0.9273 | 0.2091 | 0.4182 |
| 3->7 | 0.9352 | 0.3056 | 0.4074 |
| 4->0 | 0.9375 | 0.2344 | 0.4062 |
| 5->1 | 0.9375 | 0.1875 | 0.4250 |
| 6->2 | 0.8846 | 0.1410 | 0.2692 |
| 7->3 | 0.9333 | 0.3000 | 0.4333 |

## Held-Out Pair Accuracy, Model Seed 3

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.7250 | 0.1375 | 0.4125 |
| 1->5 | 0.6078 | 0.4020 | 0.4020 |
| 2->6 | 0.6000 | 0.1636 | 0.1909 |
| 3->7 | 0.8241 | 0.2130 | 0.2870 |
| 4->0 | 0.6875 | 0.0938 | 0.4219 |
| 5->1 | 0.7750 | 0.2000 | 0.4875 |
| 6->2 | 0.6923 | 0.1538 | 0.3333 |
| 7->3 | 0.7667 | 0.1778 | 0.2444 |

## Held-Out Pair Accuracy, Model Seed 4

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.5875 | 0.3000 | 0.3125 |
| 1->5 | 0.6863 | 0.2549 | 0.4020 |
| 2->6 | 0.6545 | 0.2091 | 0.2909 |
| 3->7 | 0.6389 | 0.3148 | 0.3611 |
| 4->0 | 0.6719 | 0.0625 | 0.2656 |
| 5->1 | 0.6000 | 0.2250 | 0.2750 |
| 6->2 | 0.6923 | 0.2821 | 0.3590 |
| 7->3 | 0.5444 | 0.2000 | 0.3778 |

The experiment uses a synthetic two-phrase relation task and frozen randomly initialized source token embeddings. The int4 values are dequantized to FP32 for training and inference. The 128-byte latent payload is larger than the estimated 8-byte 4-bit-packed payload for the original 16 token IDs, though smaller than their 2048-byte FP32 embedding input. Shared encoder/decoder storage is excluded, and no packed-runtime memory or latency savings are measured.

Each completed seed has an atomic JSON checkpoint. The aggregate JSON records checkpoint paths and SHA-256 digests, plus source hashes.
