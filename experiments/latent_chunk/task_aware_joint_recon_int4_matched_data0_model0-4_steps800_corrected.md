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
| 0 | 0.9515 | 0.8610 | 0.9941 | 0.2809 | 0.9498 | 0.3778 | 0.7416 | 0.0421 | 0.0843 |
| 1 | 0.8679 | 0.7317 | 0.9953 | 0.3090 | 0.9474 | 0.3567 | 0.5112 | 0.0590 | 0.0983 |
| 2 | 0.9642 | 0.9157 | 0.9950 | 0.2556 | 0.9716 | 0.4073 | 0.8399 | 0.0365 | 0.1236 |
| 3 | 0.8614 | 0.7079 | 0.9929 | 0.2177 | 0.9574 | 0.3371 | 0.5028 | 0.0281 | 0.1067 |
| 4 | 0.8023 | 0.6348 | 0.9911 | 0.2935 | 0.9524 | 0.3343 | 0.3315 | 0.0534 | 0.0871 |

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
| 0->4 | 0.9000 | 0.3375 | 0.4875 |
| 1->5 | 0.7647 | 0.2157 | 0.4412 |
| 2->6 | 0.8091 | 0.3364 | 0.4636 |
| 3->7 | 0.8981 | 0.2500 | 0.3056 |
| 4->0 | 0.9688 | 0.2500 | 0.3281 |
| 5->1 | 0.8000 | 0.2875 | 0.3125 |
| 6->2 | 0.8718 | 0.1923 | 0.2949 |
| 7->3 | 0.9222 | 0.3667 | 0.3556 |

## Held-Out Pair Accuracy, Model Seed 1

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.7250 | 0.1375 | 0.3625 |
| 1->5 | 0.7255 | 0.3627 | 0.4216 |
| 2->6 | 0.6273 | 0.3909 | 0.3182 |
| 3->7 | 0.8241 | 0.2963 | 0.4352 |
| 4->0 | 0.6562 | 0.3750 | 0.3594 |
| 5->1 | 0.7625 | 0.3500 | 0.4000 |
| 6->2 | 0.7179 | 0.2949 | 0.2949 |
| 7->3 | 0.8000 | 0.2444 | 0.2444 |

## Held-Out Pair Accuracy, Model Seed 2

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.9000 | 0.2750 | 0.4250 |
| 1->5 | 0.8725 | 0.3039 | 0.4510 |
| 2->6 | 0.9273 | 0.2636 | 0.4182 |
| 3->7 | 0.9352 | 0.2315 | 0.4074 |
| 4->0 | 0.9375 | 0.2188 | 0.4062 |
| 5->1 | 0.9375 | 0.1750 | 0.4250 |
| 6->2 | 0.8846 | 0.2692 | 0.2692 |
| 7->3 | 0.9333 | 0.2889 | 0.4333 |

## Held-Out Pair Accuracy, Model Seed 3

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.7250 | 0.2250 | 0.4125 |
| 1->5 | 0.6078 | 0.2157 | 0.4020 |
| 2->6 | 0.6000 | 0.1455 | 0.1909 |
| 3->7 | 0.8241 | 0.2407 | 0.2870 |
| 4->0 | 0.6875 | 0.2500 | 0.4219 |
| 5->1 | 0.7750 | 0.2875 | 0.4875 |
| 6->2 | 0.6923 | 0.2436 | 0.3333 |
| 7->3 | 0.7667 | 0.1667 | 0.2444 |

## Held-Out Pair Accuracy, Model Seed 4

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.5875 | 0.4625 | 0.3125 |
| 1->5 | 0.6863 | 0.3431 | 0.4020 |
| 2->6 | 0.6545 | 0.2455 | 0.2909 |
| 3->7 | 0.6389 | 0.2685 | 0.3611 |
| 4->0 | 0.6719 | 0.3281 | 0.2656 |
| 5->1 | 0.6000 | 0.2250 | 0.2750 |
| 6->2 | 0.6923 | 0.2821 | 0.3590 |
| 7->3 | 0.5444 | 0.2222 | 0.3778 |

The experiment uses a synthetic two-phrase relation task and frozen randomly initialized source token embeddings. The int4 values are dequantized to FP32 for training and inference. The 128-byte latent payload is larger than the estimated 8-byte 4-bit-packed payload for the original 16 token IDs, though smaller than their 2048-byte FP32 embedding input. Shared encoder/decoder storage is excluded, and no packed-runtime memory or latency savings are measured.

Each completed seed has an atomic JSON checkpoint. The aggregate JSON records checkpoint paths and SHA-256 digests, plus source hashes.

## Control Note

Both latent arms train the role and position tags and the PBSLM trunk. The frozen arm keeps the pretrained encoder fixed, while the joint arm also updates the encoder and decoder with the phrase-reconstruction loss. The historical frozen arm accidentally disabled tag gradients, so its published comparison is not a matched control.
