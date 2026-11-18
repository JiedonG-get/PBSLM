# Decoder-Free Compact Writer

Held-out modular offsets [1, 7], data seed 1.
Completed 5 of 5 model seeds.
The table shows mean +/- sample SD across model seeds on this fixed split.

| Condition | Test accuracy | Counterfactual pair exact |
|---|---:|---:|
| Raw tokens | 73.65 +/- 18.45% | 52.94 +/- 31.26% |
| Direct int4 a/b with its own trunk | 29.58 +/- 2.09% | 6.34 +/- 1.11% |
| Direct int4 a/b in the frozen teacher | 12.09 +/- 0.94% | 1.08 +/- 0.33% |
| Decoder-mediated teacher, FP32 trunk input | 96.34 +/- 1.73% | 93.24 +/- 3.02% |
| Decoder-mediated teacher, packed int4 trunk input | 95.57 +/- 1.77% | 91.76 +/- 2.96% |
| Packed writer, feature loss | 96.99 +/- 1.55% | 93.98 +/- 3.11% |
| Packed writer, feature + task losses | 90.78 +/- 13.69% | 82.93 +/- 24.36% |

Each writer starts from the same reconstruction-pretrained phrase encoder. Each phrase produces four 32-dimensional vectors from frozen token embeddings. Only training rows supply reconstruction updates, teacher supervision, distillation targets, and writer updates. The teacher stays frozen during distillation. The feature loss matches the pre-tag merged vectors. The joint arm gives task cross-entropy and feature loss equal weight and uses the same starting weights and minibatches as the feature-only arm.

The writer stores each pre-tag phrase code in 64 bytes, or 128 bytes for both phrases. It unpacks those codes, adds learned phrase/position tags, then packs and unpacks a separate 128-byte int4 trunk-input payload. The trunk computes in FP32 after unpacking. Ideal packing of the 16 synthetic token IDs uses 8 bytes. Writer accuracy is measured on a synthetic task with two simultaneously available phrases.
