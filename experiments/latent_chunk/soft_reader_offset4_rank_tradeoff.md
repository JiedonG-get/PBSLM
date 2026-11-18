# Offset-4 Soft-Reader Rank Trade-off

## Protocol

The synthetic relation task uses two 8-token phrases and counterfactual query pairs. Ordered value pairs with offset 4 were removed from each seed's original training pool before fitting the phrase autoencoder or task model. The resulting held-out row counts are seed-dependent because the source pool is sampled: seed 0 has 712 rows, seed 1 has 672, and seed 2 has 644. Each model uses 800 task-training steps, batch size 128, AdamW at 0.001, and the same seed-specific filtered split.

Each phrase is encoded independently into rank 1, 2, or 4 vectors of width 32. The fixed-scale int4-equivalent quantizer is `round(z * 7).clamp(-7, 7) / 7`. The soft reader applies the frozen phrase decoder, takes a softmax over its token logits, and projects those probabilities into the task model's trainable 10-row input vocabulary embedding table. PBSLM therefore receives 16 continuous vectors per example. The controls train on direct quantized latents, argmax-decoded token IDs, or raw token IDs.

## Rank Means for Seeds 0–2

The rank-1/2/4 comparison below uses seeds 0–2 only. Task and reconstruction figures are arithmetic means over these three seeds. The soft-reader delta is paired within each seed against that seed's raw-token baseline.

| Rank | Estimated int4 payload | Quantized token accuracy | Both phrases exact | Direct latent task | Soft-reader task | Decoded-ID task | Raw-token task | Soft minus raw |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 B | 89.34% | 19.77% | 28.02% | 38.61% | 49.37% | 86.49% | -47.88 pp |
| 2 | 64 B | 95.80% | 54.21% | 25.75% | 63.78% | 67.59% | 86.49% | -22.71 pp |
| 4 | 128 B | 99.45% | 91.92% | 27.43% | 82.39% | 86.42% | 86.49% | -4.10 pp |

The raw-token counterfactual pair-exact mean is 74.55%. Soft-reader pair-exact means are 13.40%, 40.33%, and 68.56% for ranks 1, 2, and 4. Soft reading remains below the raw-token baseline for all three seeds at each rank. Rank 4 nearly matches raw-token task accuracy on average, but does not exceed it consistently.

## Per-Seed Results

| Rank | Seed | Test rows | Int4 both-phrase exact | Soft task accuracy | Raw task accuracy | Soft pair exact | Raw pair exact | Soft-minus-raw task |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 712 | 22.19% | 42.56% | 86.10% | 15.17% | 74.16% | -43.54 pp |
| 1 | 1 | 672 | 17.86% | 32.74% | 83.78% | 9.82% | 69.05% | -51.04 pp |
| 1 | 2 | 644 | 19.25% | 40.53% | 89.60% | 15.22% | 80.43% | -49.07 pp |
| 2 | 0 | 712 | 50.00% | 60.11% | 86.10% | 34.83% | 74.16% | -25.98 pp |
| 2 | 1 | 672 | 59.67% | 63.69% | 83.78% | 39.58% | 69.05% | -20.09 pp |
| 2 | 2 | 644 | 52.95% | 67.55% | 89.60% | 46.58% | 80.43% | -22.05 pp |
| 4 | 0 | 712 | 95.79% | 85.96% | 86.10% | 73.60% | 74.16% | -0.14 pp |
| 4 | 1 | 672 | 88.99% | 74.26% | 83.78% | 55.06% | 69.05% | -9.52 pp |
| 4 | 2 | 644 | 90.99% | 86.96% | 89.60% | 77.02% | 80.43% | -2.64 pp |

## Storage and Cost

The payload estimates assume bit-packed signed 4-bit values, no scale metadata, two phrase codes, and 32 components per latent vector. The experiment itself stores and computes with dequantized FP32 values, which occupy 256 B, 512 B, or 1,024 B for ranks 1, 2, and 4. It does not test a packed kernel or runtime memory/latency savings. The soft reader expands the code to 16 continuous width-32 input vectors, or 2,048 B in FP32 per example.

The 16 synthetic input IDs use only 10 possible symbols, so a theoretical fixed-width token representation needs 4 bits per ID, or 8 B total. The 64 B raw-input figure in comparisons assumes 32-bit integers, while the current `torch.long` tensors use 128 B. Rank-4 int4 codes are estimated at 128 B, so they are not smaller than an ideally bit-packed token sequence in this benchmark.

The shared autoencoder has 25,840, 29,968, or 38,224 parameters for ranks 1, 2, and 4. The decoder alone has 7,312, 9,360, or 13,456 parameters and is required as shared side information. The soft reader adds no separate learned parameters, but reuses this decoder and routes gradients through 10 rows of the downstream model's 18-by-32 token embedding table. A 32/64/128 B code is not the full deployed system size.

This is a small synthetic task with a frozen randomly initialized source embedding table. The rank-1/2 sweep uses three seeds and the rank-4 comparison uses eight. It supports a compression-versus-reconstruction-versus-task-accuracy trade-off on this benchmark. It does not establish natural-language memory utility or actual inference savings.

## Artifacts

- Rank-4 confirmatory result: `soft_reader_int4_offset4_3seed_rank4_800.json`
- Rank-1/2 sweep result: `soft_reader_int4_offset4_rank12_3seed_800.json`
- Rank-4 seeds 3–7 extension: `soft_reader_int4_offset4_extra5seed_rank4_800.json`
- Rank-4 driver: `evaluation/latent_chunk_soft_reader_confirmatory.py`
- Rank-1/2 driver: `evaluation/latent_chunk_soft_reader_rank_sweep.py`

All three result JSON files record configuration and source hashes. All captured source hashes were verified against the current files. The experiments were run from commit `f5ae7ee1be077cc060ef88b5b1ac84695dc74ff9` with local uncommitted research artifacts.

## Rank-4 Extension on Seeds 3–7

The rank-4 confirmatory driver was run unchanged on seeds 3–7 with the same offset-4 split and 800-step configuration. Counts vary with sampled pairs. Task deltas are paired against the raw-token result for the same seed. The raw baseline is the reference with a zero delta.

| Seed | Train/test rows | Int4 token/exact reconstruction | Direct train/test | Soft train/test | Decoded-ID train/test | Raw train/test | Direct Δ | Soft Δ | Decoded Δ | Raw Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 3402/694 | 99.16%/88.04% | 99.29%/24.21% | 77.98%/48.85% | 76.40%/48.13% | 75.57%/52.59% | -28.39 pp | -3.75 pp | -4.47 pp | 0.00 pp |
| 4 | 3460/636 | 99.48%/93.40% | 99.57%/27.36% | 94.28%/84.28% | 95.55%/84.75% | 95.58%/88.21% | -60.85 pp | -3.93 pp | -3.46 pp | 0.00 pp |
| 5 | 3444/652 | 99.57%/93.40% | 99.19%/24.08% | 85.28%/63.19% | 85.60%/63.19% | 86.15%/65.95% | -41.87 pp | -2.76 pp | -2.76 pp | 0.00 pp |
| 6 | 3392/704 | 98.93%/85.65% | 99.06%/22.87% | 83.90%/60.94% | 84.35%/66.48% | 84.99%/68.89% | -46.02 pp | -7.95 pp | -2.41 pp | 0.00 pp |
| 7 | 3472/624 | 99.33%/90.71% | 99.05%/24.20% | 90.35%/75.16% | 94.76%/84.62% | 95.16%/87.98% | -63.78 pp | -12.82 pp | -3.37 pp | 0.00 pp |

## Rank-4 Combined Eight-Seed Results

The rank-4 means below combine seeds 0–7 using an unweighted arithmetic mean across seeds. Int4 reconstruction averages 99.35% token accuracy and 90.87% both-phrase exact reconstruction. Test task accuracy averages 25.63% for direct latents, 72.45% for the soft reader, 75.80% for argmax-decoded IDs, and 77.89% for raw tokens. Paired task deltas versus raw tokens average -52.26, -5.43, -2.09, and 0.00 percentage points respectively. Counterfactual pair-exact means are 4.61%, 52.27%, 56.85%, and 60.32% in the same order.

Mean train/test accuracy across the eight seeds is 99.25%/25.63% for direct latents, 89.75%/72.45% for the soft reader, 90.32%/75.80% for decoded IDs, and 90.24%/77.89% for raw tokens. The corresponding mean train-test gaps are 73.63, 17.30, 14.52, and 12.35 percentage points. Seeds 3, 5, and 6 have noticeably weaker task-training fit for soft, decoded, and raw conditions than seeds 0–2. Direct latents fit training rows near 99% in every seed but generalize poorly, indicating strong overfitting. The five-seed extension therefore mixes generalization performance with initialization-dependent optimization and fit limits.

The five-seed extension result is `soft_reader_int4_offset4_extra5seed_rank4_800.json`. It records the unchanged driver configuration and source hashes. The captured source hashes were verified against the current files.
