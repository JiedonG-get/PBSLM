# Multi-Vector Latent Chunks as PBSLM Inputs

## Conclusion

PBSLMv2 can accept learned continuous vectors as its main input sequence. In this experiment, each of `a`, `b`, `c`, and `d` was represented by 1, 2, or 4 vectors and the concatenated vectors were passed through the existing trunk. This establishes interface feasibility.

The current encoder did not preserve the raw-token model's generalization on the tested held-out relation combinations. Across three seeds, the full latent model reached 51.8% to 57.5% test accuracy by rank, compared with 84.4% for the matched raw-token baseline. Increasing the number of vectors did not produce a consistent gain. The experiment does not establish that the proposed memory mechanism is impossible, and it does not test natural-language semantics.

A follow-up removed the `c/d` construction, probes, and alignment loss entirely. With only task supervision, the separate `a(x)+b(y)` condition scored 44.8% to 48.8% across ranks, compared with 84.4% for the matched raw-token baseline. It reached high training accuracy but remained far below the raw-token baseline on held-out value pairs. See [PURE_AB_THINKING_REPORT.md](PURE_AB_THINKING_REPORT.md) for the protocol, paired results, and limitations.

A separate reconstruction-first test measured the full-phrase recovery rate, then froze the encoder and trained PBSLMv2 on `a+b`. Rank 4 reached 99.8% token accuracy and 96.4% exact recovery of both phrases, while the thinking trunk reached only 35.2% held-out task accuracy versus 84.4% for raw tokens. See [RECONSTRUCTION_THINKING_REPORT.md](RECONSTRUCTION_THINKING_REPORT.md) for full metrics and caveats.

## Method

Each example contains an 8-token query phrase `x` and an 8-token relation phrase `y`. The query names one of two keys in `y`, and the target is that key's associated value. There are eight target classes. Training uses 4,096 examples. Testing uses 3,072 examples from eight ordered value pairs reserved from training. Every class occurs in training, all complete 16-token rows are unique across the split, and adjacent test rows share `y` while querying different keys.

The shared span encoder maps `x` to `a` and `y` to `b`. A joint encoder maps the pooled pair `(a, b)` to `c` and the reversed pair `(b, a)` to `d`. Learned role and position vectors distinguish the four groups. For rank `r`, the PBSLM trunk receives `4r` continuous vectors through its `inputs_embeds` path. Raw-token inputs continue to use the existing token-ID path.

The sweep uses PBSLMv2 with `d=32`, three seeds, ranks 1, 2, and 4, 800 optimizer steps per training run, and batch size 128. Within each seed, the trunk initialization and minibatch sequence are shared across ranks and training modes. The raw-token baseline uses the same initial trunk and minibatch sequence. The separately trained `a+b` condition is reported separately from test-time ablations of the full model.

The full objective combines the target loss with field reconstruction losses for the query key and four relation symbols, a supervised target probe on pooled `c/d`, and a `c/d` alignment term. Therefore, `c/d` are not unsupervised semantic representations in this experiment. The `zero_cd` intervention zeros both the `c/d` content and their learned tags. The `mismatch_cd` intervention substitutes `c/d` from the paired example that has the same `y` and the opposite query. Mismatch is the more direct test of query-specific `c/d` information because it preserves the slot tags.

## Results

Entries are mean and sample standard deviation across three seeds. Accuracy is reported in percent. Chance accuracy is 12.5%.

| Rank | Full test | Full train | Separately trained `a+b` test | Raw-token test | Full pair exact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 57.3 +/- 6.8 | 99.2 +/- 0.5 | 82.1 +/- 11.2 | 84.4 +/- 12.9 | 32.6 +/- 7.3 |
| 2 | 51.8 +/- 4.8 | 98.5 +/- 0.4 | 68.7 +/- 20.0 | 84.4 +/- 12.9 | 26.6 +/- 8.7 |
| 4 | 57.5 +/- 11.9 | 99.0 +/- 0.3 | 59.2 +/- 21.5 | 84.4 +/- 12.9 | 32.6 +/- 18.8 |

The raw-token test values by seed are 90.6%, 69.5%, and 93.0%. The raw baseline outperformed the full latent condition in all nine matched seed/rank comparisons. The full model nearly memorized training examples while generalizing less well to the held-out value pairs.

## Fixed-Data Rank-4 Follow-up

To separate model initialization effects from data sampling, a follow-up fixed the offset-4 split generated with data seed 0 and varied model seeds 0–4. Every run used the same 3,384 training and 712 test rows. Each model seed fit one rank-4 phrase autoencoder for 800 steps, then compared raw-token, direct int4 latent, soft-reader, and argmax-decoded inputs from the same initial PBSLM state at 800 and 1,600 task-training steps. Means below are unweighted across the five model seeds. Paired differences are computed within seed against that seed's raw-token result and are reported as mean ± sample standard deviation in percentage points.

| Task steps | Direct int4 test | Soft-reader test | Decoded-ID test | Raw-token test | Direct Δ vs raw | Soft Δ vs raw | Decoded Δ vs raw |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 800 | 27.13% | 72.64% | 76.38% | 77.02% | -49.89 ± 12.58 pp | -4.38 ± 6.45 pp | -0.65 ± 2.62 pp |
| 1,600 | 27.95% | 95.11% | 95.84% | 99.19% | -71.24 ± 3.86 pp | -4.07 ± 1.92 pp | -3.34 ± 2.36 pp |

At 1,600 steps, every seed's soft-reader result remained below raw-token input. The paired soft-reader deltas at 800 steps for seeds 0–4 were -0.14, -12.22, -4.07, -9.13, and +3.65 pp. At 1,600 steps they were -3.23, -3.79, -4.49, -7.02, and -1.83 pp. Mean train accuracy was 88.85% for soft-reader and 88.95% for raw tokens at 800 steps, and 99.96% and 99.99% at 1,600 steps. Thus, the remaining test gap after near-complete training fit is not explained by incomplete task training alone.

The int4 rank-4 codes reached 99.38% mean all-token reconstruction accuracy and 91.43% mean exact recovery of both phrases across the five model seeds. The separately trained argmax-decoded condition stayed closer to raw-token performance than direct latent input. The soft-reader result shows that decoder-mediated input can approach the raw baseline, while direct trunk input from these codes generalizes poorly. These five runs share one sampled data split and only eight held-out ordered value pairs, so they estimate initialization and optimization variability conditional on that split, not variation across tasks or datasets. The result JSON and runner are [fixed_data_seed0_model_seeds0-4_rank4_budgets800_1600.json](fixed_data_seed0_model_seeds0-4_rank4_budgets800_1600.json) and [latent_chunk_fixed_data_budget.py](../../evaluation/latent_chunk_fixed_data_budget.py).

| Rank | `b` four-symbol field decode | `c/d` target probe | Full minus `zero_cd` | Full minus `mismatch_cd` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 72.3 +/- 5.2 | 46.4 +/- 3.3 | +3.7 pp | +1.8 pp |
| 2 | 57.7 +/- 2.3 | 39.3 +/- 4.3 | +8.7 pp | +2.4 pp |
| 4 | 48.0 +/- 1.9 | 37.7 +/- 2.3 | +7.2 pp | -0.4 pp |

The `a` query-key field probe reached 100% across ranks and seeds. The `b` result is a four-symbol field probe, not reconstruction of the full phrase. The target probe is trained with target labels and does not by itself demonstrate that `c/d` encode reusable meaning. Full-minus-mismatch differences are small and inconsistent, so the current results do not show robust query-specific benefit from `c/d`. The larger full-minus-zero differences should be interpreted cautiously because that intervention also removes the learned slot tags.

## Task-Aware Joint Tuning Follow-up

**Historical control correction.** In the original 800-step run, `train_frozen_int4` applied role and position tags inside `torch.no_grad()`. Those tags therefore received gradients only in the joint arm. The historical table preserves the observed measurements, but its frozen-versus-joint gap is confounded by tag training as well as encoder updates and the reconstruction objective.

The original experiment used data seed 0 and model seeds 0–4. It compared a raw-token baseline, a trunk trained on frozen rank-4 int4 codes, and a model that jointly tuned the encoder, decoder, role tags, and trunk using task cross-entropy plus full-phrase reconstruction cross-entropy with weight 1.0. Each phase ran for 800 steps. The latent codes used straight-through fake quantization and remained dequantized FP32 tensors at runtime.

| Condition | Test accuracy | Counterfactual pair exact |
|---|---:|---:|
| Raw-token baseline | 77.02 ± 11.52% | 58.54 ± 20.37% |
| Frozen int4 latent | 23.76 ± 2.85% | 4.66 ± 1.19% |
| Joint task + reconstruction | 36.26 ± 3.05% | 10.00 ± 1.60% |

The original joint arm scored 8.15–18.54 pp higher than the frozen arm across the five seeds, with an observed mean paired difference of +12.50 ± 4.02 pp. This estimate is not a matched-control effect because the frozen tags did not train.

### Matched-Tag Recheck

A corrected runner trains the role and position tags in both latent arms. Both arms start from the same pretrained encoder, model weights, tag initialization, and task minibatch sequence. The frozen arm updates only the trunk and tags. The joint arm also updates the encoder and decoder with the reconstruction objective. All five frozen-arm runs assert that both tag tensors changed. The same offset-4 split, data seed 0, and model seeds 0–4 were used at both budgets.

| Task budget per arm | Raw-token test | Matched frozen int4 test | Joint int4 test | Paired joint-minus-frozen |
|---:|---:|---:|---:|---:|
| 800 | 77.02 ± 11.52% | 27.13 ± 3.58% | 36.26 ± 3.05% | +9.13 ± 4.72 pp |
| 1,600 | 99.19 ± 0.59% | 27.95 ± 3.70% | 37.02 ± 3.25% | +9.07 ± 6.27 pp |

Values are means ± sample SD across five model seeds on the same fixed synthetic split. Joint accuracy exceeded matched frozen-latent accuracy for all five seeds at both budgets. Counterfactual pair-exact accuracy at 800 steps was 58.54 ± 20.37% for raw tokens, 4.38 ± 1.25% for matched frozen int4, and 10.00 ± 1.60% for joint int4. At 1,600 steps it was 98.37 ± 1.18%, 5.96 ± 1.84%, and 12.75 ± 3.25%, respectively.

Doubling the task budget changed the mean frozen and joint latent accuracies by less than one percentage point, while raw-token accuracy rose by 22.16 pp. This is consistent with an input-representation or interface bottleneck in this task, though it does not isolate that cause. Mean int4 all-token reconstruction was 99.39% before task tuning at both budgets, and 99.53% and 99.70% after joint tuning at 800 and 1,600 steps. Exact recovery of both phrases rose from 91.43% to 92.95% and 95.39%, respectively. High reconstruction therefore coexists with weak direct-latent relation generalization.

The corrected per-seed results, checkpoint digests, and source hashes are in [the 800-step JSON](task_aware_joint_recon_int4_matched_data0_model0-4_steps800_corrected.json) and [Markdown summary](task_aware_joint_recon_int4_matched_data0_model0-4_steps800_corrected.md), and in the [1,600-step JSON](task_aware_joint_recon_int4_matched_data0_model0-4_steps1600_corrected.json) and [Markdown summary](task_aware_joint_recon_int4_matched_data0_model0-4_steps1600_corrected.md). The corrected runner is [latent_chunk_task_aware_matched.py](../../evaluation/latent_chunk_task_aware_matched.py).

### Alternate Relation Holdout

To test whether the optimization effect persists across a different set of unseen relations, a separate balanced split held out offsets 3 and 5, covering 16 ordered value pairs. It has 4,096 training rows and 3,072 test rows, with exact target balance and no relation-pair or full-phrase `y` overlap between splits. Five model seeds ran at 800 task steps. Because this split uses a different generator and test budget from the legacy offset-4 split, interpret it as an independent replication rather than a paired split comparison.

| Condition | Test accuracy | Counterfactual pair exact |
|---|---:|---:|
| Raw-token baseline | 78.44 ± 14.22% | 60.63 ± 25.14% |
| Matched frozen int4 latent | 30.83 ± 0.97% | 6.50 ± 0.84% |
| Joint int4 latent | 40.20 ± 4.14% | 13.35 ± 4.14% |

The mean paired joint-minus-frozen difference was +9.36 ± 4.12 pp across model seeds. Joint accuracy exceeded frozen accuracy for all five seeds and on each of the 16 pair-wise seed-averaged comparisons. Raw-token accuracy remained substantially higher. The split supports the same narrow optimization finding on a second set of held-out relations, while the remaining gap shows that high phrase reconstruction alone is insufficient for direct short-code reasoning. These five seeds share one split and measure initialization/optimization variation conditional on it. The [offsets 3/5 JSON](task_aware_joint_recon_int4_matched_data0_offsets3-5_model0-4_steps800.json) and [Markdown summary](task_aware_joint_recon_int4_matched_data0_offsets3-5_model0-4_steps800.md) contain per-seed and per-pair results with checkpoint hashes.

The exact runner source for this earlier three-condition offsets 3/5 artifact is not present in Git history, although its source hash is recorded in the JSON. The later task-only offsets 3/5 run records and preserves the matched runner source used for its four-condition replication.

### Task-Only Encoder Adaptation on Balanced Splits

An ablation tested whether task loss alone could adapt the encoder, with the decoder frozen after reconstruction pretraining. All latent conditions trained the trunk and role tags from matched initializations. The joint condition trained the encoder and decoder with task loss plus reconstruction loss. Both balanced splits used 4,096 training and 3,072 test rows, rank 4 per phrase, and 800 steps per phase. Results are mean ± sample SD across five model seeds.

| Split | Raw-token test | Matched frozen int4 | Task-only int4 | Joint int4 | Task-only − frozen | Task-only − joint |
|---|---:|---:|---:|---:|---:|---:|
| Offset 4 | 85.22 ± 14.27% | 29.15 ± 1.58% | 39.47 ± 3.84% | 36.97 ± 3.41% | +10.32 ± 4.04 pp | +2.50 ± 2.83 pp |
| Offsets 3/5 | 78.44 ± 14.22% | 30.83 ± 0.97% | 40.99 ± 2.46% | 40.20 ± 4.14% | +10.16 ± 1.67 pp | +0.79 ± 3.35 pp |

Task-only exceeded the frozen condition in all five seeds on each split. Its paired advantage over joint training was less stable, with four of five positive seeds on offset 4 and three of five on offsets 3/5. With the original decoder held fixed, task-only int4 reconstruction reached 64.83% and 65.11% token accuracy, while both phrases were recovered exactly in 0.08% and 0.07% of examples. Joint training reached 99.48% and 99.67% token accuracy, with exact recovery of both phrases at 92.40% and 95.01%. These outcomes do not isolate the reconstruction loss because the joint arm also trains the decoder. The five seeds share each split, so they measure initialization and optimization variability conditional on those synthetic data splits.

Counterfactual pair-exact accuracy was 12.14 ± 2.19% for task-only, 5.81 ± 1.39% for matched frozen, and 10.53 ± 1.90% for joint training on offset 4. On offsets 3/5, the corresponding values were 13.84 ± 1.66%, 6.50 ± 0.84%, and 13.35 ± 4.14%.

The [offset-4 task-only JSON](task_aware_int4_task_only_offset4_data0_model0-4_steps800.json) and [summary](task_aware_int4_task_only_offset4_data0_model0-4_steps800.md), and the [offsets 3/5 task-only JSON](task_aware_int4_task_only_offsets3-5_data0_model0-4_steps800.json) and [summary](task_aware_int4_task_only_offsets3-5_data0_model0-4_steps800.md), include per-seed results, source hashes, and checkpoint digests. The matched runner and balanced split generator are [latent_chunk_task_aware_matched.py](../../evaluation/latent_chunk_task_aware_matched.py) and [latent_chunk_holdout.py](../../evaluation/latent_chunk_holdout.py).

## Compact Reader

The first run froze the rank-4 phrase autoencoder after 800 reconstruction steps. The compact reader independently decoded each int4 phrase code to eight soft token embeddings, merged adjacent embeddings with one shared linear layer, normalized the resulting vectors, and added learned phrase-role and slot tags. PBSLM received four vectors per phrase and learned the relation task through the trunk. The compact merge and tags add 2,080 and 512 trainable parameters, respectively. All arms used matched trunk initializations, minibatch sequences, and 800 task steps. Each fixed balanced split used 4,096 training rows, 3,072 test rows, and model seeds 0–4.

| Held-out offsets | Input | Vectors | Test accuracy | Counterfactual pair exact |
|---|---|---:|---:|---:|
| 4 | Raw tokens | 16 | 85.22 ± 14.27% | 72.79 ± 24.76% |
| 4 | Direct int4 `a/b` | 8 | 29.15 ± 1.58% | 5.81 ± 1.39% |
| 4 | Compact soft reader | 8 | 96.07 ± 2.22% | 92.63 ± 3.99% |
| 4 | Full soft reader | 16 | 80.34 ± 12.75% | 64.36 ± 21.04% |
| 3/5 | Raw tokens | 16 | 78.44 ± 14.22% | 60.63 ± 25.14% |
| 3/5 | Direct int4 `a/b` | 8 | 30.83 ± 0.97% | 6.50 ± 0.84% |
| 3/5 | Compact soft reader | 8 | 97.09 ± 0.40% | 94.52 ± 0.57% |
| 3/5 | Full soft reader | 16 | 74.58 ± 15.43% | 54.36 ± 26.51% |

Across five matched model seeds per split, the compact reader exceeded raw-token accuracy by +10.85 ± 14.30 pp on offset 4 and +18.65 ± 14.09 pp on offsets 3/5. All ten paired differences were positive. Values are means ± sample SD. The compact and full-reader arms differ in merge parameters, normalization, and explicit tags, so the first run cannot attribute the gain to the merge alone. The seeds share fixed synthetic splits with a fixed phrase layout and training budget.

The compact reader presents 1,024 B of FP32 vectors to the trunk, compared with 2,048 B for the full reader, while still computing all sixteen soft embeddings internally. The stored-code estimate is 128 B with ideal int4 packing, compared with 8 B for packed source token IDs. The experiment used simulated int4 and did not measure end-to-end memory, latency, natural-language generalization, or online state retrieval.

Per-seed results and source hashes are in the [first-run JSON](compact_reader_balanced_data0_model0-4_steps800.json) and [summary](compact_reader_balanced_data0_model0-4_steps800.md). The [historical runner](https://github.com/JiedonG-get/PBSLM/blob/d06756828a882c0b3fb7839c35e535715f5b10e3/evaluation/latent_chunk_compact_reader.py) is preserved at commit `d067568`.

### Matched-Tag Control

A follow-up added the same learned phrase-role and slot tags to the sixteen-vector soft reader. The two 16-vector arms use matched model initialization, data split, minibatch sequence, and 800 task steps. Results are mean ± sample SD across five seeds on each fixed split.

| Held-out offsets | Input | Test accuracy | Counterfactual pair exact |
|---|---|---:|---:|
| 4 | Compact, 8 vectors | 96.07 ± 2.22% | 92.63 ± 3.99% |
| 4 | Soft reader with tags, 16 vectors | 95.13 ± 2.95% | 90.99 ± 5.15% |
| 4 | Untagged soft reader, 16 vectors | 80.34 ± 12.75% | 64.36 ± 21.04% |
| 3/5 | Compact, 8 vectors | 97.09 ± 0.40% | 94.52 ± 0.57% |
| 3/5 | Soft reader with tags, 16 vectors | 96.60 ± 0.63% | 93.45 ± 1.11% |
| 3/5 | Untagged soft reader, 16 vectors | 74.58 ± 15.43% | 54.36 ± 26.51% |

The paired compact-minus-tagged accuracy difference was +0.94 ± 0.94 percentage points on offset 4 and +0.49 ± 0.45 points on offsets 3/5. Pair-exact differences were +1.64 ± 1.61 and +1.07 ± 0.82 points, respectively. Compact accuracy was higher for all five seeds on offset 4 and four of five seeds on offsets 3/5. Pair-exact accuracy was higher in all ten seed/split comparisons. The larger tagged-versus-untagged difference shows that these explicit tags account for much of the gain over the original full soft reader. The compact-vs-tagged comparison still changes pair mixing and normalization as well as trunk input length, so it does not isolate a sequence-length effect.

This remains decoder-mediated task input compression. The compact path materializes sixteen decoded embeddings before merging them to eight trunk vectors. The 128-byte int4 payload exceeds the 8-byte ideal packing of this synthetic input, and no runtime memory or latency was measured. Per-seed metrics and source hashes are in the [matched-tag JSON](compact_reader_matched_tags_data0_model0-4_steps800.json) and [summary](compact_reader_matched_tags_data0_model0-4_steps800.md).

### Packed Int4 Reader

A quantization follow-up compared the same FP32-trained compact checkpoint before and after an actual signed-nibble pack/unpack round trip, then compared it with a separate quantization-aware training arm. All arms used five matched model seeds, the same balanced splits, and 800 task steps. The post-hoc rows isolate inference quantization because both scores come from the same trained weights.

| Held-out offsets | Condition | Test accuracy | Counterfactual pair exact |
|---|---|---:|---:|
| 4 | FP32-trained, FP32 input | 96.07 ± 2.22% | 92.63 ± 3.99% |
| 4 | Same checkpoint, packed int4 input | 95.18 ± 2.40% | 90.90 ± 4.32% |
| 4 | Quantization-aware trained, packed int4 input | 95.76 ± 2.34% | 92.01 ± 4.31% |
| 3/5 | FP32-trained, FP32 input | 97.09 ± 0.40% | 94.52 ± 0.57% |
| 3/5 | Same checkpoint, packed int4 input | 96.17 ± 0.60% | 92.77 ± 0.98% |
| 3/5 | Quantization-aware trained, packed int4 input | 97.17 ± 0.56% | 94.64 ± 0.93% |

Post-hoc int4 reduced accuracy by 0.89 ± 0.51 percentage points on offset 4 and 0.92 ± 0.34 points on offsets 3/5. Pair-exact accuracy fell by 1.73 ± 1.02 and 1.74 ± 0.63 points. Every seed scored lower after post-hoc quantization. Quantization-aware training recovered 0.58 ± 0.62 and 1.00 ± 0.81 accuracy points relative to the packed post-hoc reader, with less consistent paired gains on offset 4. This supports the feasibility of feeding the task-trained compact representation through a 4-bit payload for this synthetic task.

The payload is eight tagged 32-dimensional vectors, or 128 bytes per example, compared with 1,024 bytes as FP32 vectors. It excludes shared reader and trunk parameters. The compact reader still decodes the source phrases to construct the stored vectors. The evaluation verifies nibble packing and dequantization in memory before PBSLM inference, not a serialized storage pipeline, a packed-input kernel, measured runtime memory, or latency. At 128 bytes, the latent payload remains sixteen times larger than the ideal 8-byte packing of this task's 16 source IDs. See the [int4 JSON](compact_reader_int4_data0_model0-4_steps800.json) and [summary](compact_reader_int4_data0_model0-4_steps800.md) for per-seed metrics and source hashes.

### Decoder-Free Read Adapter

A shared 128→64→128 MLP maps each packed rank-4 phrase code to four compact input vectors. The adapter receives only `a/b`. For feature distillation, its training targets are the teacher's pre-tag compact vectors. The trunk and tags stay frozen, and the three adapter objectives use the same initialization and 800 minibatch draws. At test time the adapter consumes the dequantized vectors recovered from packed `a/b` bytes and bypasses the phrase vocabulary decoder. The teacher evaluates its final input vectors in FP32, while the student packs and unpacks its tagged input vectors before the trunk.

| Held-out offsets | Condition | Test accuracy | Counterfactual pair exact |
|---|---|---:|---:|
| 4 | Raw tokens | 85.22 ± 14.27% | 72.79 ± 24.76% |
| 4 | Compact teacher | 95.93 ± 2.47% | 92.47 ± 4.41% |
| 4 | Direct `a/b` into frozen teacher trunk | 11.99 ± 1.69% | 1.00 ± 0.33% |
| 4 | Feature-distilled adapter | 85.80 ± 2.74% | 73.23 ± 4.83% |
| 4 | Task-only adapter | 35.49 ± 4.42% | 5.51 ± 1.52% |
| 4 | Feature + task adapter | 57.73 ± 13.04% | 29.79 ± 19.74% |
| 3/5 | Raw tokens | 78.44 ± 14.22% | 60.63 ± 25.14% |
| 3/5 | Compact teacher | 97.25 ± 0.46% | 94.82 ± 0.86% |
| 3/5 | Direct `a/b` into frozen teacher trunk | 12.12 ± 1.26% | 0.90 ± 0.37% |
| 3/5 | Feature-distilled adapter | 84.90 ± 3.62% | 71.81 ± 6.09% |
| 3/5 | Task-only adapter | 39.67 ± 0.95% | 6.34 ± 1.71% |
| 3/5 | Feature + task adapter | 51.57 ± 14.36% | 21.29 ± 20.70% |

Feature distillation outperformed either other adapter objective in all ten seed/split comparisons. Relative to the raw-token baseline, its paired accuracy difference was +0.58 ± 12.83 percentage points on offset 4 and +6.46 ± 14.73 points on offsets 3/5, with positive differences in two and three of five seeds, respectively. This experiment does not establish a reliable advantage over raw tokens. The adapter trailed the compact teacher by 10.13 ± 4.70 and 12.34 ± 3.46 points. Those teacher comparisons include both the changed reader and the difference in final input quantization.

The adapter adds 16,576 shared parameters and reads a 128-byte packed `a/b` record. The packed intermediate trunk input also occupies 128 bytes during this evaluation. This study uses synthetic phrases and matched splits from the earlier reader studies. It measures downstream relation accuracy and storage payload. Runtime memory, latency, and natural-language generalization were outside the evaluation. The [result JSON](compact_reader_decoder_free_data0_model0-4_steps800.json), [summary](compact_reader_decoder_free_data0_model0-4_steps800.md), and [runner](../../evaluation/latent_chunk_compact_student.py) give the per-seed results and exact source hashes.

### Decoder-Free Compact Writer

A separate confirmation used data seed 1 and held out offsets 1 and 7, which reserve sixteen ordered relation pairs for testing. All five model seeds used the same 4,096 training rows, 3,072 test rows, and 800-step budgets. Each writer starts from the reconstruction-pretrained phrase encoder, then receives frozen source-token embeddings and maps each eight-token phrase directly to four normalized 32-dimensional vectors. Feature targets come from the compact teacher's pre-tag reader output on training rows. The teacher, tags, and source embeddings remain frozen during writer training. At evaluation, the pre-tag codes are packed to int4, restored, tagged, and the trunk input is separately packed to int4 before being restored for FP32 computation.

| Condition | Test accuracy | Counterfactual pair exact |
|---|---:|---:|
| Raw tokens with separately trained trunk | 73.65 ± 18.45% | 52.94 ± 31.26% |
| Direct int4 `a/b` with separately trained trunk | 29.58 ± 2.09% | 6.34 ± 1.11% |
| Direct int4 `a/b` in the frozen compact teacher | 12.09 ± 0.94% | 1.08 ± 0.33% |
| Decoder-mediated teacher, FP32 trunk input | 96.34 ± 1.73% | 93.24 ± 3.02% |
| Same teacher, packed int4 trunk input | 95.57 ± 1.77% | 91.76 ± 2.96% |
| Decoder-free writer, feature distillation | 96.99 ± 1.55% | 93.98 ± 3.11% |
| Decoder-free writer, feature plus task loss | 90.78 ± 13.69% | 82.93 ± 24.36% |

Feature-only writers exceeded the paired raw-token model on all five seeds, by a mean of 23.35 ± 18.35 percentage points. Against the same seed's decoder-mediated teacher, the mean difference was +0.65 ± 2.74 points, with the writer higher on two seeds. Against the packed teacher it was +1.42 ± 2.67 points, with the writer higher on four seeds. The equal-weight feature-plus-task arm was less stable and scored below the FP32 teacher on four seeds. These results show that this small writer can reproduce task-useful teacher features for the held-out synthetic relations without running the phrase vocabulary decoder at read time. They do not establish a general advantage over token input.

The writer adds 24,768 shared parameters. Its persistent rank-4 int4 `a/b` payload is 128 bytes per example, and the separately packed tagged trunk input is another 128 bytes while reading. The ideal fixed-width encoding of the sixteen source IDs in this task is 8 bytes. Thus the learned representation improves this task's measured accuracy while increasing payload size. The experiment uses one data split and five model initializations, fixed phrase layout, random frozen token embeddings, and no measured runtime memory or latency. The [result JSON](compact_writer_balanced_data1_offsets1-7_model0-4_steps800.json), [summary](compact_writer_balanced_data1_offsets1-7_model0-4_steps800.md), and [runner](../../evaluation/latent_chunk_compact_writer.py) record per-seed results and provenance.

## Payload and State

The following are FP32 payload arithmetic per example. They exclude shared model and encoder parameters, metadata, allocator overhead, and any online state implementation.

| Rank | Input vectors | Four-group payload | `a+b` payload | Hypothetical one-layer K/V for 16 tokens |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 512 B | 256 B | 4,096 B |
| 2 | 8 | 1,024 B | 512 B | 4,096 B |
| 4 | 16 | 2,048 B | 1,024 B | 4,096 B |

The input contains 16 token IDs, which are 64 B when packed as 32-bit integers and 128 B in the current `torch.long` tensors. The same input is 2,048 B as 16 learned 32-dimensional FP32 embeddings. Rank 4 has the same number of input vectors and the same payload size as those embeddings. The conventional K/V arithmetic is 4,096 B per layer, or 4,096, 16,384, and 49,152 B for 1, 4, and 12 layers. These values are not measured PBSLM runtime cache sizes or memory savings. The latent encoder has 26,608, 31,024, and 39,856 trainable parameters at ranks 1, 2, and 4, including auxiliary probe heads. PBSLM has 201,306 trainable parameters in each run, so the latent condition is not parameter- or FLOP-matched.

The synthetic input alphabet has only 10 possible IDs (0–9), so an ideal fixed-width representation needs 4 bits per ID, or 8 B for the 16-token input. The 64 B comparison above assumes 32-bit integer storage and is not an optimally packed token baseline. The rank-4 int4 latent payload estimate is 128 B, which is larger than this bit-packed synthetic input before counting the shared encoder and decoder. See the [packed-state storage audit](PACKED_STATE_STORAGE_AUDIT.md) for int4 payload estimates, break-even arithmetic, and packing limitations.

An evaluation-only fixed FP32 delta state used IID random normalized keys, with 64 episodes at each length. At rank 1 and 96 stored examples, the state occupies 8,192 B versus 55,296 B for an explicit key-plus-payload table. The oldest item's mean payload cosine was 0.074, and state-query accuracy was 12.0%, compared with 57.3% when the model read the pristine payload directly. Latest-item payload cosine was 0.992. This indicates strong recency retention and substantial interference for old items in this random-key test. It is not a semantic-addressing test, and the pristine-table result is an upper-bound reference.

### Rank-4 Compact-Value State Confirmation

A five-seed evaluation tested whether the rank-4 compact `b(y)` payload could be stored in a fixed delta state and later combined with the live `a(x)` feature by the frozen compact teacher. It used data seed 1, held-out offsets 3 and 5, 4,096 training rows, 3,072 test rows, and 64 random-key episodes per tested length and seed. The state has 16 address dimensions and 128 FP32 value dimensions. Episode lengths were 8, 32, and 96. Accuracy and pair-exact scores below are means ± sample SD across model seeds.

| Stored items | Query age in later writes | Exact packed table accuracy | Fixed delta accuracy | Learned delta accuracy | Learned pair exact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 7 | 97.03 ± 1.50% | 57.19 ± 7.04% | 61.25 ± 4.90% | 40.00 ± 8.60% |
| 8 | 4 | 96.88 ± 1.83% | 71.09 ± 2.65% | 70.62 ± 7.30% | 51.25 ± 11.97% |
| 8 | 0 | 96.88 ± 1.66% | 97.19 ± 1.31% | 94.06 ± 2.57% | 89.38 ± 4.47% |
| 32 | 31 | 96.88 ± 1.10% | 15.94 ± 2.32% | 16.72 ± 1.42% | 0.94 ± 1.40% |
| 32 | 16 | 97.66 ± 1.75% | 22.97 ± 2.11% | 27.19 ± 2.24% | 7.50 ± 2.04% |
| 32 | 0 | 97.34 ± 0.43% | 96.41 ± 1.71% | 93.75 ± 2.28% | 87.81 ± 4.04% |
| 96 | 95 | 96.72 ± 1.60% | 11.88 ± 2.17% | 11.09 ± 3.24% | 0.31 ± 0.70% |
| 96 | 48 | 97.34 ± 0.70% | 13.44 ± 3.96% | 15.31 ± 5.37% | 0.62 ± 0.86% |
| 96 | 0 | 96.72 ± 1.69% | 95.94 ± 1.02% | 92.34 ± 3.05% | 85.62 ± 5.68% |

The uniform-guess reference is 12.5% task accuracy and 1.56% pair exact. The delta state retains recent values well, while accuracy for older values falls toward chance as more items are written. At 96 items, the oldest and middle queries are near chance even though the exact packed table remains near 97%. Reset-state and wrong-address controls are also near chance. These controls indicate that task performance depends on the stored state and supplied address, but the result demonstrates a recency-limited associative store rather than robust long-term retrieval. The learned condition combines a learned write strength with a trained linear read projection, so its differences from fixed delta do not isolate either component. Fixed delta scored higher on the latest query at all three lengths.

The FP32 state occupies 8,192 B per episode. An exact table stores a 16-dimensional FP32 address (64 B) and rank-4 int4 `b` payload (64 B) per item, for 1,024, 4,096, and 12,288 B at lengths 8, 32, and 96. The state reaches a payload-only break-even at 64 items and is only 1.5× smaller than the table at 96 items. The learned reader adds 16,513 shared parameters, about 66 KB in FP32, which are excluded from those per-episode counts. Feature extraction still uses the compact teacher's decoder. Addresses and write segmentation are supplied by the evaluator, so semantic addressing and autonomous online writing remain untested. The [complete five-seed result](compact_reader_state_data1_offsets3-5_model0-4.json) and [runner](../../evaluation/latent_chunk_compact_state.py) include raw metrics and source hashes.

## Scope

The data are synthetic and test held-out value-pair composition, not natural language, multimodal grounding, unseen key vocabularies, or online memory writing. Directly passing quantized `a/b` codes to a trunk trained on reader outputs performs poorly. A feature-distilled writer restores high accuracy on the tested synthetic splits without the vocabulary decoder at read time, but uses more payload bytes than ideal source-token packing. The random-key state test does not evaluate learned semantic addressing. These results support the feasibility of trained continuous multi-vector inputs for this narrow task, while practical memory efficiency, robust open-domain composition, and the `a/b/c/d` state mechanism remain unproven.

Raw per-run outputs and source provenance are in [formal_matched_3seed_rank_800.json](formal_matched_3seed_rank_800.json).
