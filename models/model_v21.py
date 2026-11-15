"""v2.1 单路径连续向量基线模型（Baseline A）。

设计约束（见 PLAN_v2_1.md）：
- 单路径自回归预测；
- 连续向量输出，不做全词表 softmax；
- 不使用 A/C 双路径、InfoFlow、E/M、PerturbField、Shock/Activity；
- 输出码本支持：
  * trainable：独立可训练码本（与输入 embedding 不共享）；
  * fixed：固定随机单位码本（固定 seed，训练中不被优化器更新）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model import BlockBank
except ImportError:
    from model import BlockBank


class SinglePathContinuousLM(nn.Module):
    def __init__(self, vocab=16384, d=256, heads=8, k=8, depth=2,
                 codebook="trainable", codebook_seed=1234):
        super().__init__()
        assert codebook in ("trainable", "fixed")
        self.vocab, self.d, self.k, self.codebook = vocab, d, k, codebook
        self.tok = nn.Embedding(vocab, d)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.bank = BlockBank(k, depth, d, heads)
        # Baseline A 不调用 BlockBank.comm()，冻结这些参数，避免“可训练但前向不用”
        for _m in (self.bank.comm1, self.bank.comm2):
            for _p in _m.parameters():
                _p.requires_grad_(False)
        self.bank.comm_beta.requires_grad_(False)
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)
        # A1：独立可训练码本，不与输入 embedding（self.tok）共享参数
        self.code_table = nn.Embedding(vocab, d)
        nn.init.normal_(self.code_table.weight, std=0.02)

        if codebook == "fixed":
            g = torch.Generator().manual_seed(int(codebook_seed))
            with torch.no_grad():
                self.code_table.weight.copy_(
                    F.normalize(torch.randn(vocab, d, generator=g), dim=-1))
            self.code_table.weight.requires_grad_(False)

    def target_embed(self, ids):
        return F.normalize(self.code_table.weight[ids.long()], dim=-1)

    @torch.no_grad()
    def decode_similarity(self, z):
        zz = F.normalize(z.float(), dim=-1)
        E = F.normalize(self.code_table.weight.float(), dim=-1)
        return zz @ E.t()

    def forward(self, ids):
        """ids: (B,T) -> z: (B,d) 连续向量。"""
        x = self.tok(ids.long())
        Y = self.bank.block_forward(x, torch.arange(self.k, device=ids.device))
        delta = Y - x.unsqueeze(0)
        h = self.ln(x + delta.mean(0))
        ctx = h[:, -1] + h.mean(1)          # 最后 token + 全文上下文
        z = F.normalize(self.out(ctx), dim=-1)
        return z
