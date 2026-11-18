"""Run paired post-commit text correction with and without the spoken prefix."""

import argparse
import asyncio
import json
from pathlib import Path

from evaluation.local_qwen_adapter import LocalQwenAdapter
from evaluation.provenance import capture
from evaluation.qwen_scheduler_smoke import LocalQwenBackend
from runtime.continuous import ContinuousRunner


SCENARIOS = (
    ('inspection', '维修组确认，钟楼例行检修从上午九点开始。钟楼例行检修几点开始？',
     '登记台放了两把备用伞。', '维修组发来新通知，钟楼例行检修改到下午三点开始。'),
    ('handoff', '陶艺工坊通知，烧制编号七的作品由沈禾签收。谁签收编号七的作品？',
     '展柜外的走廊刚刚打扫过。', '工坊公布最新取件安排，编号七的作品改由程芮签收。'),
    ('room', '场务确认，周五诗歌讲座使用青和厅。请问周五的诗歌讲座在哪个厅？',
     '签到桌上增加了一份节目单。', '场务刚发布通知，周五诗歌讲座改在晓竹厅。'),
)
SYSTEM = '你正在参与一段对话。根据最新输入自然作答，采用明确确认的新信息。每次用一句简短的话回应。'


async def run_scenario(model, scenario, include_prefix, token_budget):
    name, first, background, correction = scenario

    def render(request):
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': request.inputs[0][1]}]
        if include_prefix and request.committed_prefix:
            messages.append({'role': 'assistant', 'content': request.committed_prefix})
        if len(request.inputs) > 1:
            messages.append({'role': 'user', 'content': '\n'.join(text for _, text in request.inputs[1:])})
        return messages

    backend = LocalQwenBackend(model, token_budget, render_messages=render)
    runner = ContinuousRunner(backend)

    async def wait_for_revision(revision):
        return await runner.wait_event(
            lambda event: event['kind'] in {'generation_finished', 'generation_failed'}
            and event['generation_revision'] == revision, timeout=90,
        )

    try:
        initial_revision = await runner.observe(first)
        await wait_for_revision(initial_revision)
        initial = runner.output
        background_revision = await runner.observe_background(
            background, relevance_assertion='Scripted unrelated observation',
        )
        background_preserved = runner.output == initial
        correction_revision = await runner.observe(correction)
        await wait_for_revision(correction_revision)
    finally:
        try:
            await runner.close(timeout=90)
        finally:
            await backend.close()

    repair_units = [item for item in runner.committed if item['revision'] == correction_revision]
    return {
        'scenario': name,
        'prefix_condition': 'present' if include_prefix else 'omitted',
        'initial_output': initial,
        'repair_output': ''.join(item['unit'] for item in repair_units),
        'initial_commit_preserved': bool(initial) and runner.output.startswith(initial),
        'background_preserved_output': bool(initial) and background_preserved,
        'background_started_generation': any(event['kind'] == 'generation_started'
                                             and event['generation_revision'] == background_revision
                                             for event in runner.journal),
        'generation_errors': [event for event in runner.journal if event['kind'] == 'generation_failed'],
        'backend_metrics': backend.metrics,
        'journal': runner.journal,
        'committed': runner.committed,
    }


async def execute(model, token_budget):
    return [await run_scenario(model, scenario, include_prefix, token_budget)
            for scenario in SCENARIOS for include_prefix in (True, False)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new-tokens', default=48, type=int)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        parser.error('generation token budget must be in 1..256')
    if args.output.exists():
        raise FileExistsError(args.output)
    model = LocalQwenAdapter(args.model_dir)
    report = {
        'scope': 'scripted query, background and correction; whole-answer commits; post-commit semantics annotated separately',
        'checkpoint_files_sha256': model.file_hashes,
        'runs': asyncio.run(execute(model, args.max_new_tokens)),
        'provenance': capture(['evaluation/local_qwen_adapter.py', 'evaluation/qwen_scheduler_smoke.py',
                               'evaluation/qwen_postcommit_repair.py', 'runtime/continuous.py']),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    if any(run['generation_errors'] for run in report['runs']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
