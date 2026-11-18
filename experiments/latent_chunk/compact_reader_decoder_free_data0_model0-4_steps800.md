# Decoder-Free Compact Reader

- Status: true (10 completed runs)
- Model seeds: [0, 1, 2, 3, 4]
- Student task steps: 800
- Results are mean ± sample SD across model seeds on each fixed balanced split.

| Held-out offsets | Input condition | Test accuracy | Pair exact |
|---|---|---:|---:|
| [3, 5] | Raw-token baseline | 78.44 ± 14.22% | 60.63 ± 25.14% |
| [3, 5] | Decoder-mediated compact teacher | 97.25 ± 0.46% | 94.82 ± 0.86% |
| [3, 5] | Direct int4 `a/b` codes | 12.12 ± 1.26% | 0.90 ± 0.37% |
| [3, 5] | Feature-distilled int4 adapter | 84.90 ± 3.62% | 71.81 ± 6.09% |
| [3, 5] | Task-trained int4 adapter | 39.67 ± 0.95% | 6.34 ± 1.71% |
| [3, 5] | Feature + task int4 adapter | 51.57 ± 14.36% | 21.29 ± 20.70% |
| [4] | Raw-token baseline | 85.22 ± 14.27% | 72.79 ± 24.76% |
| [4] | Decoder-mediated compact teacher | 95.93 ± 2.47% | 92.47 ± 4.41% |
| [4] | Direct int4 `a/b` codes | 11.99 ± 1.69% | 1.00 ± 0.33% |
| [4] | Feature-distilled int4 adapter | 85.80 ± 2.74% | 73.23 ± 4.83% |
| [4] | Task-trained int4 adapter | 35.49 ± 4.42% | 5.51 ± 1.52% |
| [4] | Feature + task int4 adapter | 57.73 ± 13.04% | 29.79 ± 19.74% |

The student reads only the stored rank-4 int4 `a/b` vectors. A shared 128→64→128 MLP maps each phrase code to four compact trunk vectors. Feature-distillation targets come from the compact teacher's pre-tag merger outputs on training examples. Feature-only, task-only, and combined objectives are compared with matched adapter initialization and minibatch draws. The compact teacher uses the phrase decoder during reading, while student evaluation does not.

Evaluation packs the student's eight tagged 32-dimensional vectors into 128 bytes and unpacks them before the frozen PBSLM trunk. The adapter adds 16,576 shared parameters. Source-phrase encoding still requires the existing encoder when writing a memory. This is a synthetic relation-composition benchmark and does not demonstrate language generalization, end-to-end storage savings, or online memory writing.
