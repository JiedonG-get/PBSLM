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

The next experiment used the same data seed 0 split and model seeds 0–4. For each seed, it compared a raw-token baseline, a trunk trained on frozen rank-4 int4 codes, and a model that jointly tuned the encoder, decoder, role tags, and trunk using task cross-entropy plus full-phrase reconstruction cross-entropy with weight 1.0. Each phase ran for 800 steps. The latent codes used straight-through fake quantization and remained dequantized FP32 tensors at runtime.

| Condition | Test accuracy | Counterfactual pair exact |
|---|---:|---:|
| Raw-token baseline | 77.02 ± 11.52% | 58.54 ± 20.37% |
| Frozen int4 latent | 23.76 ± 2.85% | 4.66 ± 1.19% |
| Joint task + reconstruction | 36.26 ± 3.05% | 10.00 ± 1.60% |

The joint model improved on frozen latents for all five seeds by 8.15–18.54 pp, with a mean paired gain of 12.50 ± 4.02 pp. It remained below the raw-token baseline for every seed, with a mean paired difference of -40.76 ± 8.62 pp. Averaged across the five seeds, all eight held-out ordered value pairs improved over the frozen-latent model, yet joint accuracy remained far below raw-token accuracy for each pair.

Joint tuning preserved phrase reconstruction. Mean int4 all-token accuracy rose from 99.39 ± 0.24% before task tuning to 99.53 ± 0.13% afterward, and mean both-phrase exact recovery rose from 91.43 ± 3.14% to 92.95 ± 1.82%. High reconstruction accuracy therefore did not translate into comparable held-out relation reasoning. This is a promising optimization effect on one synthetic split, not evidence of natural-language memory capability or a general solution. The split contains only eight held-out ordered value pairs, and all five model seeds reuse that same split. The per-seed and per-pair measurements, checkpoint digests, and source hashes are in [task_aware_joint_recon_int4_offset4_data0_model0-4.json](task_aware_joint_recon_int4_offset4_data0_model0-4.json) and [its Markdown summary](task_aware_joint_recon_int4_offset4_data0_model0-4.md).

## Payload and State

The following are FP32 payload arithmetic per example. They exclude shared model and encoder parameters, metadata, allocator overhead, and any online state implementation.

| Rank | Input vectors | Four-group payload | `a+b` payload | Hypothetical one-layer K/V for 16 tokens |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 512 B | 256 B | 4,096 B |
| 2 | 8 | 1,024 B | 512 B | 4,096 B |
| 4 | 16 | 2,048 B | 1,024 B | 4,096 B |

The input contains 16 token IDs, which are 64 B when packed as 32-bit integers and 128 B in the current `torch.long` tensors. The same input is 2,048 B as 16 learned 32-dimensional FP32 embeddings. Rank 4 has the same number of input vectors and the same payload size as those embeddings. The conventional K/V arithmetic is 4,096 B per layer, or 4,096, 16,384, and 49,152 B for 1, 4, and 12 layers. These values are not measured PBSLM runtime cache sizes or memory savings. The latent encoder has 26,608, 31,024, and 39,856 trainable parameters at ranks 1, 2, and 4, including auxiliary probe heads. PBSLM has 201,306 trainable parameters in each run, so the latent condition is not parameter- or FLOP-matched.

The synthetic input alphabet has only 10 possible IDs (0–9), so an ideal fixed-width representation needs 4 bits per ID, or 8 B for the 16-token input. The 64 B comparison above assumes 32-bit integer storage and is not an optimally packed token baseline. The rank-4 int4 latent payload estimate is 128 B, which is larger than this bit-packed synthetic input before counting the shared encoder and decoder.

An evaluation-only fixed FP32 delta state used IID random normalized keys, with 64 episodes at each length. At rank 1 and 96 stored examples, the state occupies 8,192 B versus 55,296 B for an explicit key-plus-payload table. The oldest item's mean payload cosine was 0.074, and state-query accuracy was 12.0%, compared with 57.3% when the model read the pristine payload directly. Latest-item payload cosine was 0.992. This indicates strong recency retention and substantial interference for old items in this random-key test. It is not a semantic-addressing test, and the pristine-table result is an upper-bound reference.

## Scope

The data are synthetic and test held-out value-pair composition, not natural language, multimodal grounding, unseen key vocabularies, or online memory writing. The auxiliary probes add supervision and parameters to the latent condition. The random-key state test does not evaluate learned semantic retrieval. Results support the claim that multi-vector continuous inputs can run through the PBSLM trunk, while the particular `a/b/c/d` construction needs better compositional generalization before it can support a practical memory claim.

Raw per-run outputs and source provenance are in [formal_matched_3seed_rank_800.json](formal_matched_3seed_rank_800.json).
