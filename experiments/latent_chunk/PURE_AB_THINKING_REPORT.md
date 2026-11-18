# Task-Only a+b Thinking Experiment

## Question

Can the PBSLM thinking trunk combine separate compact representations of the query phrase `x` and relation phrase `y` without an explicit pair-combination representation such as `c` or `d`?

## Method

The synthetic task gives `x` a query key and gives `y` two key-value edges. The target is the value associated with the queried key. Training has 4,096 examples and testing has 3,072. Testing uses eight ordered pairs of edge values absent from training while retaining every individual class. Adjacent examples share the same `y` and query its two different keys.

A shared span encoder maps `x` to `a(x)` and `y` to `b(y)`. Each is represented by rank 1, 2, or 4 vectors. Learned role and position vectors mark the ordered `a` and `b` groups, which are concatenated and passed to PBSLMv2 through `inputs_embeds`. The trunk receives 2, 4, or 8 vectors. The model does not construct `c/d`, and training uses only the relation-query target loss. There are no content reconstruction heads, pair probes, or alignment losses.

The raw-token baseline uses the same PBSLM trunk initialization, target loss, batch order, optimizer settings, and training steps for each seed. Each seed is run at all three ranks. The training uses 800 AdamW steps with batch size 128 and learning rate 0.001. Reported values are mean and sample standard deviation across three seeds. Paired deltas are computed per seed as latent accuracy minus raw-token accuracy.

## Results

Accuracy is reported in percent. Chance accuracy is 12.5%.

| Rank | Latent test | Raw-token test | Paired difference | Latent train | Raw-token train | Latent pair exact | Raw-token pair exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48.8 +/- 1.1 | 84.4 +/- 12.9 | -35.5 +/- 14.1 pp | 87.6 +/- 1.4 | 91.3 +/- 6.6 | 19.8 +/- 0.6 | 70.9 +/- 23.0 |
| 2 | 47.1 +/- 0.8 | 84.4 +/- 12.9 | -37.2 +/- 13.2 pp | 94.4 +/- 0.6 | 91.3 +/- 6.6 | 17.8 +/- 0.2 | 70.9 +/- 23.0 |
| 4 | 44.8 +/- 0.6 | 84.4 +/- 12.9 | -39.5 +/- 12.4 pp | 97.5 +/- 0.4 | 91.3 +/- 6.6 | 17.1 +/- 1.4 | 70.9 +/- 23.0 |

The raw-token baseline won all nine seed/rank comparisons. More latent vectors did not improve held-out pair accuracy. Rank 4 fitted the training examples most closely and had the lowest test accuracy. The strong train-test gap and low pair-exact score indicate that the trunk does not reliably compose the two independently encoded inputs on this split.

This result does not show that internal composition is impossible. It shows that the tested PBSLMv2 setup, trained with this target loss and these compact encoders, did not learn the same generalization as the raw-token path.

## Footprint and Limits

The latent payload alone is 256, 512, or 1,024 bytes per example in FP32 at ranks 1, 2, and 4. This excludes the shared encoder, metadata, and runtime overhead. The encoder has 18,656, 20,864, or 25,280 trainable parameters. The trunk has 201,306 trainable parameters. The baseline receives 16 token IDs, while the latent trunk receives 2, 4, or 8 vectors. The comparison is matched for trunk initialization and optimization schedule, not for parameter count, sequence length, or FLOPs.

The encoders are learned from the synthetic task, and the input is explicitly divided into `x` and `y`. Learned role and position vectors preserve that known ordering. The result therefore does not test discovering the roles of arbitrary spans. The split tests composition of held-out ordered value pairs, not unseen key-value bindings, natural-language reasoning, online memory writing, or multimodal semantics. Three seeds provide a preliminary estimate rather than a broad scaling result.

## Artifact

Raw per-run outputs and source hashes are in [pure_ab_thinking_3seed_rank_800.json](pure_ab_thinking_3seed_rank_800.json). The runnable experiment is [latent_chunk_ab_thinking.py](../../evaluation/latent_chunk_ab_thinking.py).
