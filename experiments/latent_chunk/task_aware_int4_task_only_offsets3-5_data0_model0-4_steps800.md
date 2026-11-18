# Task-Aware Joint Reconstruction Experiment

- Status: `complete` (5/5 model seeds completed)
- Data seed: `0`
- Model seeds requested: `[0, 1, 2, 3, 4]`
- Split: ordered value pairs at offsets [3, 5] held out before training (4096 train, 3072 test rows)
- Rank: `4` per phrase, with `8` input vectors total
- Steps: `800` reconstruction pretraining, `800` frozen-latent trunk training, `800` joint tuning, `800` raw baseline
- Objective: task cross-entropy + `1.0` x full-phrase reconstruction cross-entropy
- Quantization: signed int4-equivalent fake quantization, with FP32 tensors and straight-through gradients

## Task Results

| Model seed | Raw train | Raw test | Frozen train | Frozen test | Joint train | Joint test | Raw pair-exact | Frozen pair-exact | Joint pair-exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8667 | 0.7454 | 0.9834 | 0.3184 | 0.9434 | 0.3766 | 0.5430 | 0.0690 | 0.1094 |
| 1 | 0.7131 | 0.5837 | 0.9846 | 0.3102 | 0.9473 | 0.4749 | 0.2467 | 0.0671 | 0.2070 |
| 2 | 0.9504 | 0.9258 | 0.9961 | 0.3141 | 0.9395 | 0.3949 | 0.8555 | 0.0703 | 0.1204 |
| 3 | 0.8413 | 0.7493 | 0.9812 | 0.3060 | 0.9373 | 0.3825 | 0.5475 | 0.0684 | 0.1113 |
| 4 | 0.9705 | 0.9180 | 0.9890 | 0.2930 | 0.9268 | 0.3809 | 0.8385 | 0.0501 | 0.1191 |

## Int4 Reconstruction

| Model seed | Before task tuning, token acc. | Before, both phrases exact | After joint tuning, token acc. | After, both phrases exact |
|---:|---:|---:|---:|---:|
| 0 | 0.9954 | 0.9333 | 0.9973 | 0.9583 |
| 1 | 0.9953 | 0.9313 | 0.9973 | 0.9577 |
| 2 | 0.9946 | 0.9202 | 0.9968 | 0.9525 |
| 3 | 0.9964 | 0.9486 | 0.9961 | 0.9417 |
| 4 | 0.9956 | 0.9326 | 0.9960 | 0.9401 |

## Held-Out Pair Accuracy, Model Seed 0

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->3 | 0.7812 | 0.3021 | 0.1979 |
| 0->5 | 0.7604 | 0.4948 | 0.5104 |
| 1->4 | 0.7604 | 0.1406 | 0.3594 |
| 1->6 | 0.7812 | 0.2708 | 0.3177 |
| 2->5 | 0.7552 | 0.3802 | 0.3542 |
| 2->7 | 0.7708 | 0.3802 | 0.4062 |
| 3->0 | 0.7812 | 0.2604 | 0.1927 |
| 3->6 | 0.7396 | 0.2500 | 0.4271 |
| 4->1 | 0.7552 | 0.2760 | 0.3385 |
| 4->7 | 0.7240 | 0.3906 | 0.3906 |
| 5->0 | 0.6875 | 0.4531 | 0.5208 |
| 5->2 | 0.7396 | 0.3385 | 0.3438 |
| 6->1 | 0.6823 | 0.2604 | 0.3906 |
| 6->3 | 0.6927 | 0.2708 | 0.4375 |
| 7->2 | 0.7552 | 0.3542 | 0.3958 |
| 7->4 | 0.7604 | 0.2708 | 0.4427 |

## Held-Out Pair Accuracy, Model Seed 1

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->3 | 0.5417 | 0.4010 | 0.5885 |
| 0->5 | 0.6198 | 0.3750 | 0.4688 |
| 1->4 | 0.5208 | 0.3438 | 0.6146 |
| 1->6 | 0.5781 | 0.2448 | 0.3750 |
| 2->5 | 0.6198 | 0.2604 | 0.4167 |
| 2->7 | 0.5104 | 0.3021 | 0.4479 |
| 3->0 | 0.5677 | 0.2812 | 0.5104 |
| 3->6 | 0.5104 | 0.2656 | 0.5573 |
| 4->1 | 0.6354 | 0.2292 | 0.4688 |
| 4->7 | 0.6458 | 0.4323 | 0.4219 |
| 5->0 | 0.6042 | 0.4688 | 0.5260 |
| 5->2 | 0.5677 | 0.3333 | 0.4219 |
| 6->1 | 0.5573 | 0.2083 | 0.4115 |
| 6->3 | 0.5885 | 0.2448 | 0.4740 |
| 7->2 | 0.5990 | 0.2865 | 0.3958 |
| 7->4 | 0.6719 | 0.2865 | 0.5000 |

## Held-Out Pair Accuracy, Model Seed 2

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->3 | 0.9531 | 0.3125 | 0.3438 |
| 0->5 | 0.9427 | 0.3021 | 0.3490 |
| 1->4 | 0.9271 | 0.2448 | 0.3281 |
| 1->6 | 0.9531 | 0.2031 | 0.3333 |
| 2->5 | 0.9271 | 0.3542 | 0.5000 |
| 2->7 | 0.9062 | 0.4167 | 0.3177 |
| 3->0 | 0.9010 | 0.2135 | 0.3594 |
| 3->6 | 0.8854 | 0.3698 | 0.4635 |
| 4->1 | 0.9479 | 0.3750 | 0.4062 |
| 4->7 | 0.9271 | 0.4635 | 0.4167 |
| 5->0 | 0.9010 | 0.3333 | 0.4219 |
| 5->2 | 0.9323 | 0.2917 | 0.4896 |
| 6->1 | 0.8958 | 0.1823 | 0.3958 |
| 6->3 | 0.9271 | 0.2760 | 0.4323 |
| 7->2 | 0.9583 | 0.3438 | 0.4219 |
| 7->4 | 0.9271 | 0.3438 | 0.3385 |

## Held-Out Pair Accuracy, Model Seed 3

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->3 | 0.7135 | 0.3125 | 0.4688 |
| 0->5 | 0.6406 | 0.2448 | 0.4010 |
| 1->4 | 0.7708 | 0.4375 | 0.3542 |
| 1->6 | 0.8333 | 0.2135 | 0.2240 |
| 2->5 | 0.6823 | 0.3333 | 0.4583 |
| 2->7 | 0.7760 | 0.3125 | 0.4844 |
| 3->0 | 0.8073 | 0.3125 | 0.5052 |
| 3->6 | 0.8073 | 0.2344 | 0.3490 |
| 4->1 | 0.8229 | 0.3438 | 0.3229 |
| 4->7 | 0.7969 | 0.3073 | 0.3542 |
| 5->0 | 0.7552 | 0.1719 | 0.2917 |
| 5->2 | 0.7292 | 0.3177 | 0.4688 |
| 6->1 | 0.7292 | 0.3125 | 0.2552 |
| 6->3 | 0.6927 | 0.2865 | 0.3854 |
| 7->2 | 0.7135 | 0.3854 | 0.5052 |
| 7->4 | 0.7188 | 0.3698 | 0.2917 |

## Held-Out Pair Accuracy, Model Seed 4

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->3 | 0.9479 | 0.2448 | 0.3438 |
| 0->5 | 0.9792 | 0.2708 | 0.1615 |
| 1->4 | 0.9688 | 0.2552 | 0.4375 |
| 1->6 | 0.9479 | 0.2396 | 0.3021 |
| 2->5 | 0.9427 | 0.3698 | 0.3906 |
| 2->7 | 0.9010 | 0.3385 | 0.3333 |
| 3->0 | 0.8646 | 0.3542 | 0.4583 |
| 3->6 | 0.7969 | 0.3125 | 0.4479 |
| 4->1 | 0.9427 | 0.3021 | 0.5052 |
| 4->7 | 0.9115 | 0.2396 | 0.3698 |
| 5->0 | 0.9583 | 0.2188 | 0.2188 |
| 5->2 | 0.9583 | 0.3750 | 0.4375 |
| 6->1 | 0.9479 | 0.2448 | 0.4427 |
| 6->3 | 0.7656 | 0.1875 | 0.4167 |
| 7->2 | 0.9115 | 0.3646 | 0.3490 |
| 7->4 | 0.9427 | 0.3698 | 0.4792 |

The experiment uses a synthetic two-phrase relation task and frozen randomly initialized source token embeddings. The int4 values are dequantized to FP32 for training and inference. The 128-byte latent payload is larger than the estimated 8-byte 4-bit-packed payload for the original 16 token IDs, though smaller than their 2048-byte FP32 embedding input. Shared encoder/decoder storage is excluded, and no packed-runtime memory or latency savings are measured.

Each completed seed has an atomic JSON checkpoint. The aggregate JSON records checkpoint paths and SHA-256 digests, plus source hashes.

## Control Note

All latent arms train the role and position tags and the PBSLM trunk. The frozen arm keeps the pretrained encoder fixed. The task-only arm updates only the encoder with task loss, while the joint arm also updates the encoder and decoder with phrase-reconstruction loss. The historical frozen arm accidentally disabled tag gradients, so its published comparison is not a matched control.

## Task-Only Encoder Adaptation

| Model seed | Task-only test | Joint test | Task-only minus joint | Task-only phrase exact | Joint phrase exact |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.4215 | 0.3766 | +0.0449 | 0.0003 | 0.9583 |
| 1 | 0.4339 | 0.4749 | -0.0410 | 0.0000 | 0.9577 |
| 2 | 0.4238 | 0.3949 | +0.0290 | 0.0010 | 0.9525 |
| 3 | 0.3968 | 0.3825 | +0.0143 | 0.0020 | 0.9417 |
| 4 | 0.3734 | 0.3809 | -0.0075 | 0.0000 | 0.9401 |
