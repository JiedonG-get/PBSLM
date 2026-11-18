# Task-Aware Joint Reconstruction Experiment

- Status: `complete` (5/5 model seeds completed)
- Data seed: `0`
- Model seeds requested: `[0, 1, 2, 3, 4]`
- Split: offset-4 ordered value pairs held out before training (3384 train, 712 test rows)
- Rank: `4` per phrase, with `8` input vectors total
- Steps: `800` reconstruction pretraining, `1600` frozen-latent trunk training, `1600` joint tuning, `1600` raw baseline
- Objective: task cross-entropy + `1.0` x full-phrase reconstruction cross-entropy
- Quantization: signed int4-equivalent fake quantization, with FP32 tensors and straight-through gradients

## Task Results

| Model seed | Raw train | Raw test | Frozen train | Frozen test | Joint train | Joint test | Raw pair-exact | Frozen pair-exact | Joint pair-exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 0.9986 | 0.9861 | 0.2949 | 0.9829 | 0.3862 | 0.9972 | 0.0702 | 0.1067 |
| 1 | 0.9997 | 0.9846 | 0.9950 | 0.3048 | 0.9811 | 0.3596 | 0.9691 | 0.0562 | 0.1236 |
| 2 | 1.0000 | 0.9972 | 0.9950 | 0.2640 | 0.9849 | 0.4017 | 0.9944 | 0.0506 | 0.1573 |
| 3 | 1.0000 | 0.9902 | 0.9914 | 0.2219 | 0.9775 | 0.3848 | 0.9803 | 0.0365 | 0.1629 |
| 4 | 1.0000 | 0.9888 | 0.9967 | 0.3118 | 0.9855 | 0.3188 | 0.9775 | 0.0843 | 0.0871 |

## Int4 Reconstruction

| Model seed | Before task tuning, token acc. | Before, both phrases exact | After joint tuning, token acc. | After, both phrases exact |
|---:|---:|---:|---:|---:|
| 0 | 0.9971 | 0.9579 | 0.9981 | 0.9691 |
| 1 | 0.9934 | 0.9059 | 0.9960 | 0.9368 |
| 2 | 0.9918 | 0.8933 | 0.9971 | 0.9565 |
| 3 | 0.9915 | 0.8806 | 0.9972 | 0.9593 |
| 4 | 0.9954 | 0.9340 | 0.9967 | 0.9480 |

## Held-Out Pair Accuracy, Model Seed 0

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.9875 | 0.2750 | 0.4250 |
| 1->5 | 1.0000 | 0.3922 | 0.4804 |
| 2->6 | 1.0000 | 0.4364 | 0.5091 |
| 3->7 | 1.0000 | 0.1852 | 0.2778 |
| 4->0 | 1.0000 | 0.1719 | 0.4531 |
| 5->1 | 1.0000 | 0.3375 | 0.3750 |
| 6->2 | 1.0000 | 0.2821 | 0.3718 |
| 7->3 | 1.0000 | 0.2222 | 0.2000 |

## Held-Out Pair Accuracy, Model Seed 1

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.9750 | 0.1750 | 0.3125 |
| 1->5 | 1.0000 | 0.4510 | 0.4412 |
| 2->6 | 0.9545 | 0.3455 | 0.3727 |
| 3->7 | 1.0000 | 0.2778 | 0.3333 |
| 4->0 | 1.0000 | 0.2656 | 0.3750 |
| 5->1 | 1.0000 | 0.3000 | 0.5125 |
| 6->2 | 0.9615 | 0.2821 | 0.3718 |
| 7->3 | 0.9889 | 0.2889 | 0.1667 |

## Held-Out Pair Accuracy, Model Seed 2

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 1.0000 | 0.3000 | 0.3750 |
| 1->5 | 0.9902 | 0.2745 | 0.4412 |
| 2->6 | 1.0000 | 0.3182 | 0.4182 |
| 3->7 | 1.0000 | 0.2870 | 0.3889 |
| 4->0 | 1.0000 | 0.2812 | 0.3438 |
| 5->1 | 1.0000 | 0.1500 | 0.4000 |
| 6->2 | 0.9872 | 0.1923 | 0.2949 |
| 7->3 | 1.0000 | 0.2778 | 0.5111 |

## Held-Out Pair Accuracy, Model Seed 3

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 1.0000 | 0.1625 | 0.4125 |
| 1->5 | 0.9804 | 0.2941 | 0.4216 |
| 2->6 | 0.9909 | 0.1273 | 0.1455 |
| 3->7 | 1.0000 | 0.2963 | 0.4259 |
| 4->0 | 1.0000 | 0.2031 | 0.5469 |
| 5->1 | 0.9750 | 0.2875 | 0.4875 |
| 6->2 | 0.9872 | 0.1795 | 0.2949 |
| 7->3 | 0.9889 | 0.2111 | 0.4333 |

## Held-Out Pair Accuracy, Model Seed 4

| Ordered value pair | Raw-token | Frozen int4 | Joint int4 |
|---|---:|---:|---:|
| 0->4 | 0.9875 | 0.3875 | 0.2625 |
| 1->5 | 1.0000 | 0.3529 | 0.4020 |
| 2->6 | 0.9909 | 0.3273 | 0.2727 |
| 3->7 | 0.9907 | 0.3148 | 0.4815 |
| 4->0 | 0.9844 | 0.2656 | 0.2500 |
| 5->1 | 1.0000 | 0.2750 | 0.2000 |
| 6->2 | 0.9615 | 0.3846 | 0.3077 |
| 7->3 | 0.9889 | 0.1778 | 0.3000 |

The experiment uses a synthetic two-phrase relation task and frozen randomly initialized source token embeddings. The int4 values are dequantized to FP32 for training and inference. The 128-byte latent payload is larger than the estimated 8-byte 4-bit-packed payload for the original 16 token IDs, though smaller than their 2048-byte FP32 embedding input. Shared encoder/decoder storage is excluded, and no packed-runtime memory or latency savings are measured.

Each completed seed has an atomic JSON checkpoint. The aggregate JSON records checkpoint paths and SHA-256 digests, plus source hashes.

## Control Note

Both latent arms train the role and position tags and the PBSLM trunk. The frozen arm keeps the pretrained encoder fixed, while the joint arm also updates the encoder and decoder with the phrase-reconstruction loss. The historical frozen arm accidentally disabled tag gradients, so its published comparison is not a matched control.
