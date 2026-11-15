# Training

- `train_v21.py`：Phase 0/1 使用的配置驱动训练脚本（InfoNCE、checkpoint/resume、完整评估）。
- `train_v2.py`：v2 完整模型的实验训练脚本。

从仓库根目录运行：

```bash
python -m training.train_v21 --config configs/formal_a1.json
```
