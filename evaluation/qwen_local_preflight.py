"""Check direct answers from the fixed local Qwen3 checkpoint."""
import argparse
import json
from pathlib import Path
import time

from evaluation.local_qwen_adapter import LocalQwenAdapter
from evaluation.provenance import capture


PROMPTS = [
    ('arithmetic-smoke', '请只回答一个数字：2加2等于几？', '4', None),
    ('story-current-plan',
     '林舟原本说会在车站等许岚。街边商店正在播放音乐。林舟收到消息后改口：不去车站了，到图书馆门口等。'
     '许岚的围巾是蓝色的。问题：林舟最后准备在哪里等人？请只回答地点。',
     '图书馆门口', '车站'),
    ('story-object-holder',
     '周禾拿着仓库的钥匙。周禾把钥匙交给顾南，顾南接过后放进口袋。门口有一把旧雨伞。'
     '问题：现在谁拿着那把钥匙？请只回答名字。',
     '顾南', '周禾'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        parser.error('generation token budget must be in 1..256')
    if args.output.exists():
        raise FileExistsError(args.output)

    started = time.perf_counter()
    model = LocalQwenAdapter(args.model_dir)
    load_seconds = time.perf_counter() - started
    results = []
    for prompt_id, prompt, expected, obsolete in PROMPTS:
        messages = [{'role': 'user', 'content': prompt}]
        started = time.perf_counter()
        answer, metrics = model.generate(messages, max_new_tokens=args.max_new_tokens)
        passed = (answer == expected if obsolete is None else expected in answer and obsolete not in answer)
        results.append({
            'id': prompt_id, 'messages': messages, 'expected': expected, 'answer': answer,
            'passed': passed and not metrics['thinking_detected'],
            'metrics': metrics, 'elapsed_seconds': time.perf_counter() - started,
        })

    result = {
        'scope': 'fixed local Qwen3 direct-answer smoke on three scripted prompts',
        'checkpoint_name': model.path.name,
        'checkpoint_files_sha256': model.file_hashes,
        'load_seconds': load_seconds,
        'prompts': results,
        'passed': all(case['passed'] for case in results),
        'provenance': capture(['evaluation/local_qwen_adapter.py', 'evaluation/qwen_local_preflight.py']),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({
        'passed': result['passed'],
        'prompts': [{key: case[key] for key in ('id', 'answer', 'passed')} for case in results],
    }, ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
