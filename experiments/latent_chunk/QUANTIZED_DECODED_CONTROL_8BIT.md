# 8-bit Decoded-Latent Control

## Protocol

For each seed, a shared phrase autoencoder is trained on 4,096 rows. Each phrase is encoded independently into `rank` unit-normalized 32-dimensional vectors. The codes are quantized with the fixed signed scale `round(z * 127).clamp(-127, 127) / 127`, decoded to token IDs with `argmax`, and only then passed to PBSLMv2. The trunk is trained with the query loss for 800 steps at batch size 128. The raw-token baseline uses the same seed, initial trunk state, data order, and training schedule. Test performance is measured on 3,072 rows containing ordered value pairs withheld from training.

This control tests whether the quantized codes retain information that a shared decoder can turn back into the original token sequence. It does **not** test direct reasoning on the continuous codes. PBSLM receives 16 decoded token IDs, not the short latent sequence.

## Results

The entries below are percentages. The paired delta is decoded-latent task accuracy minus the raw-token accuracy for the same seed. Values are copied from [decoded_latent_int8_thinking_3seed_rank_800.json](decoded_latent_int8_thinking_3seed_rank_800.json).

| Rank | Seed | Packed code payload | Token accuracy | Both phrases exact |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 64 B | 89.687097 | 16.210938 |
| 1 | 1 | 64 B | 91.434735 | 28.906250 |
| 1 | 2 | 64 B | 89.298505 | 21.451823 |
| 2 | 0 | 128 B | 97.900391 | 73.470056 |
| 2 | 1 | 128 B | 98.136395 | 76.822919 |
| 2 | 2 | 128 B | 98.319501 | 78.157556 |
| 4 | 0 | 256 B | 99.735516 | 95.996094 |
| 4 | 1 | 256 B | 99.800622 | 97.070312 |
| 4 | 2 | 256 B | 99.743652 | 96.256512 |

| Rank | Seed | Decoded-latent task accuracy | Raw-token accuracy | Paired delta | Decoded pair exact | Raw pair exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 42.838544 | 90.625000 | -47.786456 pp | 18.489584 | 81.835938 |
| 1 | 1 | 40.071616 | 69.498700 | -29.427084 pp | 14.713542 | 44.531250 |
| 1 | 2 | 52.701825 | 92.968750 | -40.266925 pp | 30.143231 | 86.393231 |
| 2 | 0 | 82.617188 | 90.625000 | -8.007812 pp | 67.968750 | 81.835938 |
| 2 | 1 | 62.695312 | 69.498700 | -6.803387 pp | 37.565106 | 44.531250 |
| 2 | 2 | 88.053387 | 92.968750 | -4.915363 pp | 77.604169 | 86.393231 |
| 4 | 0 | 88.964844 | 90.625000 | -1.660156 pp | 78.841150 | 81.835938 |
| 4 | 1 | 68.912762 | 69.498700 | -0.585938 pp | 44.270834 | 44.531250 |
| 4 | 2 | 92.252606 | 92.968750 | -0.716144 pp | 85.091150 | 86.393231 |

| Rank | Mean token accuracy | Mean both-phrase exact | Mean task accuracy | Mean raw accuracy | Mean paired delta | Mean pair exact / raw |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 90.140112% | 22.189670% | 45.203995% | 84.364150% | -39.160155 pp | 21.115452% / 70.920140% |
| 2 | 98.118762% | 76.150177% | 77.788629% | 84.364150% | -6.575521 pp | 61.046008% / 70.920140% |
| 4 | 99.759930% | 96.440973% | 83.376737% | 84.364150% | -0.987413 pp | 69.401045% / 70.920140% |

## Interpretation and Limits

The rank-4 codes reconstruct both eight-token phrases exactly on 96.44% of test rows on average. After decoding, task accuracy is within 0.99 percentage points of raw-token input on average, with a negative paired delta for all three seeds. Rank 2 has substantially lower task accuracy, and rank 1 loses much more. This shows that the code-plus-decoder pipeline preserves useful information on this synthetic held-out-combination task. It does not show that PBSLM can think directly over the compressed vectors.

Payload is arithmetic for two phrases, each represented by `rank × 32` signed 8-bit values, assuming bit-packed storage and no scale metadata. The original 16 IDs occupy 128 B as `torch.long` or 64 B as packed 32-bit IDs. Thus rank 1 uses half the `torch.long` input payload but loses substantial accuracy; rank 2 matches the `torch.long` payload; rank 4 uses twice that payload. The shared encoder and decoder, decoder compute, and the 16 IDs supplied to PBSLM are excluded from latent payload. The decoder is required for the tested result, so these numbers do not establish an end-to-end memory or runtime saving.

The task has explicit ordered `x/y` spans and a synthetic vocabulary. The source token embeddings are frozen random initialization. Three seeds provide limited evidence, and all runs use the same 800-step schedule. Results do not establish natural-language generalization, open-ended memory recall, or multimodal capability.

The JSON records commit `f5ae7ee1be077cc060ef88b5b1ac84695dc74ff9`, a dirty-tree snapshot, Python 3.13.5, PyTorch 2.11.0+cu128, and an RTX 4060 Laptop GPU. All seven recorded selected-source SHA-256 hashes match the current files. GPU bitwise determinism is not promised.
