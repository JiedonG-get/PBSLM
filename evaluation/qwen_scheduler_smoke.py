"""Exercise the continuous scheduler with a fixed local Qwen3 checkpoint."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path
import time

from evaluation.local_qwen_adapter import LocalQwenAdapter
from evaluation.provenance import capture
from runtime.continuous import ContinuousRunner

ROOT = Path(__file__).resolve().parents[1]
INITIAL_QUESTION = (
    '林舟原本在车站等许岚。临时计划仍是在车站等。'
    '如果计划不变，请用一句话回答林舟在哪里等许岚。'
)
CORRECTION = '计划更正：林舟不去车站了，改在图书馆门口等许岚。'
LATEST_QUESTION = '请回答：林舟最后准备在哪里等许岚？只给地点。'


class LocalQwenBackend:
    def __init__(self, model, max_new_tokens):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='qwen-inference')
        self.started = asyncio.Event()
        self.metrics = []

    def _infer(self, request, loop):
        started = time.perf_counter()
        loop.call_soon_threadsafe(self.started.set)
        messages = [
            {'role': 'system', 'content': '根据截至目前的事件回答最后一个问题，采用明确更正后的信息。只给简短答案。'},
            {'role': 'user', 'content': '\n'.join(text for _, text in request.inputs)},
        ]
        answer, metrics = self.model.generate(messages, max_new_tokens=self.max_new_tokens)
        finished = time.perf_counter()
        self.metrics.append({
            'revision': request.revision,
            'start_perf': started,
            'finish_perf': finished,
            'duration': finished - started,
            **metrics,
            'answer': answer,
        })
        return answer

    async def generate(self, request):
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(self.executor, self._infer, request, loop)
        if answer:
            yield answer

    async def close(self):
        await asyncio.to_thread(partial(self.executor.shutdown, cancel_futures=True))


async def execute(model, max_new_tokens):
    backend = LocalQwenBackend(model, max_new_tokens)
    runner = ContinuousRunner(backend, max_committed_chars=20000)
    arrivals = []

    async def arrive(text, should_respond):
        started = time.perf_counter()
        revision = await runner.observe(text, should_respond=should_respond)
        arrivals.append({
            'revision': revision,
            'start_perf': started,
            'finish_perf': time.perf_counter(),
            'text': text,
            'should_respond': should_respond,
        })
        return revision

    try:
        first_revision = await arrive(INITIAL_QUESTION, True)
        await asyncio.wait_for(backend.started.wait(), timeout=90)
        correction_revision = await arrive(CORRECTION, False)
        final_revision = await arrive(LATEST_QUESTION, True)
        await runner.wait_event(
            lambda event: event['kind'] in {'generation_finished', 'generation_failed'}
            and event['generation_revision'] == final_revision,
            timeout=90,
        )
    finally:
        try:
            await runner.close(timeout=90)
        finally:
            await backend.close()

    first = next((item for item in backend.metrics if item['revision'] == first_revision), None)
    final = next((item for item in backend.metrics if item['revision'] == final_revision), None)
    correction = next(item for item in arrivals if item['revision'] == correction_revision)
    checks = {
        'correction_arrived_during_inference': bool(
            first and first['start_perf'] <= correction['start_perf'] < first['finish_perf']
        ),
        'correction_returned_during_inference': bool(
            first and correction['finish_perf'] < first['finish_perf']
        ),
        'obsolete_generation_cancelled': any(
            event['kind'] == 'generation_cancelled' and event['generation_revision'] == first_revision
            for event in runner.journal
        ),
        'worker_serialized': bool(first and final and first['finish_perf'] <= final['start_perf']),
        'no_stale_generation_commits': bool(runner.committed) and all(
            item['revision'] == final_revision for item in runner.committed
        ),
        'committed_current_answer': '图书馆门口' in runner.output and '车站' not in runner.output,
    }
    return {
        'passed': all(checks.values()),
        **checks,
        'backend_metrics': backend.metrics,
        'arrivals': arrivals,
        'runtime_journal': runner.journal,
        'committed': runner.committed,
        'note': 'One commit per completed answer. Cancellation fences host output; the worker drains active inference.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= args.max_new_tokens <= 256:
        parser.error('generation token budget must be in 1..256')

    model = LocalQwenAdapter(args.model_dir)
    result = asyncio.run(execute(model, args.max_new_tokens))
    report = {
        'scope': 'fixed Qwen3 scheduling and current-answer smoke on one scripted scenario',
        'checkpoint_files_sha256': model.file_hashes,
        'result': result,
        'provenance': capture([
            'evaluation/local_qwen_adapter.py', 'evaluation/qwen_scheduler_smoke.py', 'runtime/continuous.py',
        ]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({
        'passed': result['passed'],
        'committed': result['committed'],
        'checks': {key: value for key, value in result.items() if type(value) is bool},
    }, ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
