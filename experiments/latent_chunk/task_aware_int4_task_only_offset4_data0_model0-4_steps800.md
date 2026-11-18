# Task-Aware Joint Reconstruction Experiment

- Status: `complete` (5/5 model seeds completed)
- Data seed: `0`
- Model seeds requested: `[0, 1, 2, 3, 4]`
- Split: ordered value pairs at offsets [4] held out before training (4096 train, 3072 test rows)
- Rank: `4` per phrase, with `8` input vectors total
- Steps: `800` reconstruction pretraining, `800` frozen-latent trunk training, `800` joint tuning, `800` raw baseline
- Objective: task cross-entropy + `1.0` x full-phrase reconstruction cross-entropy
- Quantization: signed int4-equivalent fake quantization, with FP32 tensors and straight-through gradients

## Task Results

| Model seed | Raw train | Raw test | Frozen train | Frozen test | Joint train | Joint test | Raw pair-exact | Frozen pair-exact | Joint pair-exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9739 | 0.9342 | 0.9917 | 0.2926 | 0.9363 | 0.3415 | 0.8691 | 0.0553 | 0.0827 |
| 1 | 0.7041 | 0.6045 | 0.9912 | 0.3158 | 0.9431 | 0.4004 | 0.3008 | 0.0729 | 0.1283 |
| 2 | 0.9707 | 0.9346 | 0.9846 | 0.2933 | 0.9502 | 0.4128 | 0.8770 | 0.0632 | 0.1217 |
| 3 | 0.9260 | 0.8545 | 0.9854 | 0.2741 | 0.9355 | 0.3444 | 0.7214 | 0.0358 | 0.0964 |
| 4 | 0.9702 | 0.9333 | 0.9785 | 0.2816 | 0.9500 | 0.3493 | 0.8711 | 0.0632 | 0.0977 |

## Int4 Reconstruction

| Model seed | Before task tuning, token acc. | Before, both phrases exact | After joint tuning, token acc. | After, both phrases exact |
|---:|---:|---:|---:|---:|
| 0 | 0.9977 | 0.9661 | 0.9962 | 0.9421 |
| 1 | 0.9943 | 0.9202 | 0.9954 | 0.9336 |
| 2 | 0.9943 | 0.9150 | 0.9957 | 0.9359 |
| 3 | 0.9886 | 0.8392 | 0.9910 | 0.8708 |
| 4 | 0.9944 | 0.9199 | 0.9958 | 0.9378 |

## Held-Out Pair Accuracy, Model Seed 0

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.9323 | 0.2630 | 0.3932 |
| 1->5 | 0.9271 | 0.2708 | 0.4323 |
| 2->6 | 0.9635 | 0.3438 | 0.2995 |
| 3->7 | 0.9427 | 0.3568 | 0.1797 |
| 4->0 | 0.9688 | 0.2448 | 0.3672 |
| 5->1 | 0.9141 | 0.2604 | 0.4036 |
| 6->2 | 0.9062 | 0.3411 | 0.4297 |
| 7->3 | 0.9193 | 0.2604 | 0.2266 |

## Held-Out Pair Accuracy, Model Seed 1

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.5938 | 0.3698 | 0.4740 |
| 1->5 | 0.5651 | 0.3646 | 0.4427 |
| 2->6 | 0.5729 | 0.3307 | 0.2552 |
| 3->7 | 0.6146 | 0.3438 | 0.5208 |
| 4->0 | 0.6328 | 0.2630 | 0.4896 |
| 5->1 | 0.6068 | 0.1953 | 0.3255 |
| 6->2 | 0.6172 | 0.3385 | 0.2422 |
| 7->3 | 0.6328 | 0.3203 | 0.4531 |

## Held-Out Pair Accuracy, Model Seed 2

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.9401 | 0.2656 | 0.4635 |
| 1->5 | 0.9089 | 0.2839 | 0.4870 |
| 2->6 | 0.9271 | 0.2995 | 0.4297 |
| 3->7 | 0.9479 | 0.2161 | 0.2995 |
| 4->0 | 0.9349 | 0.4089 | 0.4531 |
| 5->1 | 0.9401 | 0.3333 | 0.4714 |
| 6->2 | 0.9323 | 0.3411 | 0.4271 |
| 7->3 | 0.9453 | 0.1979 | 0.2708 |

## Held-Out Pair Accuracy, Model Seed 3

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.8802 | 0.2500 | 0.4688 |
| 1->5 | 0.7917 | 0.2995 | 0.4531 |
| 2->6 | 0.8958 | 0.1797 | 0.1328 |
| 3->7 | 0.8333 | 0.3359 | 0.3021 |
| 4->0 | 0.8125 | 0.2344 | 0.4323 |
| 5->1 | 0.9010 | 0.2760 | 0.4792 |
| 6->2 | 0.8568 | 0.3255 | 0.1432 |
| 7->3 | 0.8646 | 0.2917 | 0.3438 |

## Held-Out Pair Accuracy, Model Seed 4

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.9427 | 0.2318 | 0.4635 |
| 1->5 | 0.9141 | 0.2578 | 0.3307 |
| 2->6 | 0.9323 | 0.2422 | 0.2578 |
| 3->7 | 0.9323 | 0.4193 | 0.3359 |
| 4->0 | 0.9193 | 0.1562 | 0.4714 |
| 5->1 | 0.9453 | 0.2760 | 0.2292 |
| 6->2 | 0.9401 | 0.2839 | 0.3490 |
| 7->3 | 0.9401 | 0.3854 | 0.3568 |

The experiment uses a synthetic two-phrase relation task and frozen randomly initialized source token embeddings. The int4 values are dequantized to FP32 for training and inference. The 128-byte latent payload is larger than the estimated 8-byte 4-bit-packed payload for the original 16 token IDs, though smaller than their 2048-byte FP32 embedding input. Shared encoder/decoder storage is excluded, and no packed-runtime memory or latency savings are measured.

Each completed seed has an atomic JSON checkpoint. The aggregate JSON records checkpoint paths and SHA-256 digests, plus source hashes.

## Control Note

All latent arms train the role and position tags and the PBSLM trunk. The frozen arm keeps the pretrained encoder fixed. The task-only arm updates only the encoder with task loss, while the joint arm also updates the encoder and decoder with phrase-reconstruction loss. The historical frozen arm accidentally disabled tag gradients, so its published comparison is not a matched control.

## Task-Only Encoder Adaptation

| Model seed | Task-only test | Joint test | Task-only minus joint | Task-only phrase exact | Joint phrase exact |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.3802 | 0.3415 | +0.0387 | 0.0007 | 0.9421 |
| 1 | 0.3773 | 0.4004 | -0.0231 | 0.0003 | 0.9336 |
| 2 | 0.4632 | 0.4128 | +0.0505 | 0.0007 | 0.9359 |
| 3 | 0.3770 | 0.3444 | +0.0326 | 0.0016 | 0.8708 |
| 4 | 0.3757 | 0.3493 | +0.0264 | 0.0010 | 0.9378 |
