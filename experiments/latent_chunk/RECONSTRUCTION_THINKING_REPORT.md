# a/b Reconstruction and Thinking Test

## Method

The encoder receives each 8-token phrase independently and compresses it into `rank` vectors of width 32. A shared decoder is trained to reconstruct every token in each phrase. The encoder and decoder train on the 4,096 training rows only. Reconstruction is evaluated on 3,072 held-out rows, including ordered value pairs absent from training.

After reconstruction training, the encoder is frozen. PBSLMv2 receives only the concatenation of `a(x)` and `b(y)` through `inputs_embeds` and is trained on the relation-query target. No `c/d` codes or reconstruction decoder are used by the thinking trunk. Trainable role and position vectors identify the ordered x/y chunks. A raw-token baseline uses the same trunk initialization, query objective, optimizer schedule, and minibatch order. Each phase runs for 800 steps with batch size 128. Values below are mean and sample standard deviation across three seeds.

The reconstruction head predicts all eight token positions in each phrase. Token accuracy measures position-wise recovery. Phrase exact requires every token in that phrase to be correct. Both-phrases exact requires all 16 tokens to match. The first token of each phrase is a fixed marker, so the table also reports accuracy over variable positions and task-relevant fields.

## Results

Accuracy is reported in percent. Task chance accuracy is 12.5%.

| Rank per phrase | Total latent vectors | Payload bytes FP32 | All-token accuracy | Both phrases exact | x variable tokens | y variable tokens | Query key | Four y edge symbols | Thinking test | Thinking train | Raw-token test |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 256 | 90.1 +/- 1.1 | 22.2 +/- 6.2 | 92.6 +/- 1.2 | 85.0 +/- 2.3 | 91.8 +/- 1.6 | 78.2 +/- 4.6 | 25.3 +/- 2.6 | 96.9 +/- 0.7 | 84.4 +/- 12.9 |
| 2 | 4 | 512 | 98.1 +/- 0.2 | 76.0 +/- 2.4 | 98.36 +/- 0.01 | 97.32 +/- 0.48 | 97.9 +/- 0.2 | 96.4 +/- 0.7 | 29.0 +/- 3.8 | 98.4 +/- 0.5 | 84.4 +/- 12.9 |
| 4 | 8 | 1,024 | 99.76 +/- 0.04 | 96.4 +/- 0.6 | 99.85 +/- 0.03 | 99.59 +/- 0.06 | 99.8 +/- 0.1 | 99.36 +/- 0.14 | 35.2 +/- 3.0 | 98.1 +/- 0.6 | 84.4 +/- 12.9 |

The frozen-latent thinking trunk beat chance on the held-out task, but remained well below raw tokens in all nine seed/rank comparisons. Its counterfactual pair-exact accuracy was 5.2%, 7.0%, and 9.4% at ranks 1, 2, and 4, compared with 70.9% for raw tokens. The trunk reached 96.9% to 98.4% training accuracy, so its test results show a generalization problem rather than an inability to fit the training task.

The test indicates that high reconstructability is not enough to make the current thinking trunk reason well from the latents. Rank 4 recovers both full phrases exactly on 96.4% of test rows, yet reaches only 35.2% task accuracy. Rank 1 has 90.1% token accuracy but only 22.2% exact recovery of both phrases. These token metrics depend on the trained decoder and are not a direct measure of how interpretable the vectors are without a decoder.

## Limits

The source token embedding table is frozen and randomly initialized. The task and vocabulary are synthetic, and x/y spans are given explicitly. The experiment tests a frozen reconstruction-trained encoder followed by a separately trained thinking trunk. It does not test pretrained language semantics or joint task-aware encoder adaptation.

The encoder-decoder has 25,840, 29,968, or 38,224 parameters at ranks 1, 2, and 4. The latent payload is not smaller than the original token IDs in this setup. Two rank-1 vectors occupy 256 bytes in FP32, while the 16 input IDs occupy 128 bytes as `torch.long` or 64 bytes as int32. Payload arithmetic excludes the shared encoder and other runtime state. The reported 201,306 trunk parameters requiring gradients include 576 token-embedding parameters that the `inputs_embeds` path bypasses.

## Artifacts

Raw per-seed results and source hashes are in [reconstruction_thinking_3seed_rank_800.json](reconstruction_thinking_3seed_rank_800.json). The runnable experiment is [latent_chunk_reconstruction.py](../../evaluation/latent_chunk_reconstruction.py).
