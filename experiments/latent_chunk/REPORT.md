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

## Scope

The data are synthetic and test held-out value-pair composition, not natural language, multimodal grounding, unseen key vocabularies, or online memory writing. The auxiliary probes add supervision and parameters to the latent condition. The random-key state test does not evaluate learned semantic retrieval. Results support the claim that multi-vector continuous inputs can run through the PBSLM trunk, while the particular `a/b/c/d` construction needs better compositional generalization before it can support a practical memory claim.

Raw per-run outputs and source provenance are in [formal_matched_3seed_rank_800.json](formal_matched_3seed_rank_800.json).
