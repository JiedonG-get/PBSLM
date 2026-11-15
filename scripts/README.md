# Scripts

- `data/`：BPE 训练、语料 tokenize、语料下载/解析。
- `analysis/`：历史分析/评估脚本（v12–v15），部分依赖旧路径，主要用于回看实验。

数据准备示例：

```bash
python scripts/data/train_bpe.py --text <corpus.txt> --out tokenizer/bpe16k_tokenizer.json
python scripts/data/tokenize_corpus.py --text <corpus.txt> --out-prefix data/bpe_full \
  --tokenizer tokenizer/bpe16k_tokenizer.json
```
