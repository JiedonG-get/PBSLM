"""Compare memory-readout routing under scripted observation arrival conditions."""

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import time

from evaluation.bounded_online_memory import ARMS, DEFAULT_LABELS, make_memory, parse_query
from runtime.continuous import ContinuousRunner


CAPACITY = 16
MAX_NEW_TOKENS = 48
WAIT_SECONDS = 90
UNKNOWN = '不确定'
SYSTEM = (
    '你根据当前已确认的资料回答样件的存放位置。没有资料时回答“不确定”。'
    '如果你此前给出的地点与当前资料冲突，请明确撤回先前的说法并给出新的地点。'
    '请用一句简短的话回答。'
)


@dataclass(frozen=True)
class Case:
    name: str
    setup: tuple[str, ...]
    question: str
    arrival: str
    update: str | None = None


@dataclass(frozen=True)
class Snapshot:
    question: str
    value: str | None


def confirm(key, value):
    return f'确认：{key}的存放位置是{value}。'


def ask(key):
    return f'请问{key}的当前存放位置？'


CASES = (
    Case('precommit_room', (confirm('场务甲一号样件', '松间厅'),), ask('场务甲一号样件'),
         'during_held_first_generation', confirm('场务甲一号样件', '雨燕厅')),
    Case('precommit_cabinet', (confirm('制版室乙二号样件', '澄光柜'),), ask('制版室乙二号样件'),
         'during_held_first_generation', confirm('制版室乙二号样件', '竹溪站')),
    Case('unrelated_during_generation', (confirm('版画社丙三号样件', '石桥厅'),), ask('版画社丙三号样件'),
         'during_held_first_generation', confirm('版画社庚一号样件', '澄光柜')),
    Case('postcommit_door', (confirm('陶土坊丁一号样件', '江柳门'),), ask('陶土坊丁一号样件'),
         'after_first_generation_finished', confirm('陶土坊丁一号样件', '月湾站')),
    Case('postcommit_studio', (confirm('录音棚戊二号样件', '东练功房'),), ask('录音棚戊二号样件'),
         'after_first_generation_finished', confirm('录音棚戊二号样件', '松间厅')),
    Case('cold_retention',
         (confirm('借阅站己三号样件', '雨燕厅'),) + tuple(
             confirm(f'工艺社货架第{index:02}号样件', DEFAULT_LABELS[index % len(DEFAULT_LABELS)])
             for index in range(24)
         ), ask('借阅站己三号样件'), 'query_only'),
)

# Only the offline scorer reads these labels.
ORACLES = {
    'precommit_room': ('松间厅', '雨燕厅'),
    'precommit_cabinet': ('澄光柜', '竹溪站'),
    'unrelated_during_generation': ('石桥厅', '石桥厅'),
    'postcommit_door': ('江柳门', '月湾站'),
    'postcommit_studio': ('东练功房', '松间厅'),
    'cold_retention': ('雨燕厅', '雨燕厅'),
}


class HoldFirstOutput:
    """Gate delivery of the initial answer after the worker starts its first job."""

    def __init__(self, backend):
        self.backend = backend
        self.held_revision = None
        self.release = asyncio.Event()

    async def generate(self, request):
        async for answer in self.backend.generate(request):
            if request.revision == self.held_revision:
                await self.release.wait()
            yield answer


async def run_case(model, case, arm, *, backend_factory=None, memory_factory=make_memory):
    if backend_factory is None:
        from evaluation.qwen_scheduler_smoke import LocalQwenBackend

        backend_factory = LocalQwenBackend
    memory = memory_factory(arm, capacity=CAPACITY)
    snapshots = {}

    def render(request):
        snapshot = snapshots[request.revision]
        messages = [{'role': 'system', 'content': SYSTEM}]
        if request.committed_prefix:
            messages.extend((
                {'role': 'user', 'content': snapshot.question},
                {'role': 'assistant', 'content': request.committed_prefix},
            ))
        evidence = snapshot.value if snapshot.value is not None else '暂无已确认的地点'
        messages.append({'role': 'user', 'content': f'{snapshot.question}\n当前资料：{evidence}。'})
        return messages

    backend = backend_factory(model, MAX_NEW_TOKENS, render_messages=render)
    delivery = HoldFirstOutput(backend)
    runner = ContinuousRunner(delivery, history_limit=1)
    trace = []

    async def wait_finished(revision):
        return await runner.wait_event(
            lambda event: event['kind'] in {'generation_finished', 'generation_failed'}
            and event['generation_revision'] == revision, timeout=WAIT_SECONDS,
        )

    async def arrive(text, *, question=None, previous_value=None, is_query=False):
        received = time.perf_counter()
        revision = runner.revision + 1
        value = previous_value
        if is_query:
            question = text
            value = memory.answer(question)
            route = 'query'
        else:
            memory.observe(text)
            if question is not None:
                value = memory.peek(question)
            route = 'replace' if question is not None and value != previous_value else 'background'

        route_started = time.perf_counter()
        if route == 'background':
            await runner.observe_background(
                text, relevance_assertion='active readout unchanged or no active query',
            )
        else:
            snapshots[revision] = Snapshot(question, value)
            await runner.observe(text)
        trace.append({
            'revision': revision, 'text': text, 'kind': 'query' if is_query else 'observation',
            'active_key': parse_query(question) if question else None,
            'value_before': previous_value, 'value_after': value, 'route': route,
            'received_perf': received, 'route_started_perf': route_started,
            'memory_and_parse_seconds': route_started - received,
        })
        return revision, value, route

    initial = None
    update = None
    prefix_at_update = None
    hold_released = None
    try:
        for text in case.setup:
            await arrive(text)
        if case.arrival == 'during_held_first_generation':
            delivery.held_revision = runner.revision + 1
        initial, initial_value, _ = await arrive(case.question, is_query=True)
        if case.update is None or case.arrival == 'after_first_generation_finished':
            await wait_finished(initial)
        else:
            await asyncio.wait_for(backend.started.wait(), timeout=WAIT_SECONDS)
        if case.update is not None:
            prefix_at_update = runner.output
            update, _, route = await arrive(case.update, question=case.question, previous_value=initial_value)
            if delivery.held_revision is not None:
                hold_released = time.perf_counter()
                delivery.release.set()
            await wait_finished(update if route == 'replace' else initial)
    finally:
        try:
            await runner.close(timeout=WAIT_SECONDS)
        finally:
            await backend.close()

    first_worker = next((job for job in backend.metrics if job['revision'] == initial), None)
    update_received = trace[-1]['received_perf'] if update is not None else None
    return {
        'arm': arm, 'case': case.name, 'arrival_condition': case.arrival,
        'initial_revision': initial, 'update_revision': update,
        'held_revision': delivery.held_revision, 'hold_released_perf': hold_released,
        'initial_worker_active_at_update': (
            first_worker['start_perf'] <= update_received < first_worker['finish_perf']
            if first_worker is not None and update_received is not None else None
        ),
        'prefix_at_update': prefix_at_update,
        'prefix_preserved': runner.output.startswith(prefix_at_update) if prefix_at_update else None,
        'first_commit': runner.committed[0] if runner.committed else None,
        'committed': runner.committed, 'route_trace': trace, 'journal': runner.journal,
        'worker_jobs': backend.metrics, 'memory_metrics': memory.metrics(),
        'generation_errors': [event for event in runner.journal if event['kind'] == 'generation_failed'],
    }


def score_run(run):
    before, after = ORACLES[run['case']]
    update = run['update_revision']
    commits = [event for event in run['journal'] if event['kind'] == 'output_committed']
    scored = []
    for event in commits:
        after_update = update is not None and event['latest_observation_revision'] >= update
        expected = after if after_update else before
        labels = [value for value in DEFAULT_LABELS if value in event['unit']]
        scored.append({
            'generation_revision': event['generation_revision'], 'after_update': after_update,
            'expected': expected, 'labels': labels, 'current_label_present': expected in labels,
            'only_current_label': labels == [expected], 'uncertain_text_present': UNKNOWN in event['unit'],
            'outdated_label_present': after_update and before != after and before in labels,
            'only_outdated_label': after_update and before != after and labels == [before],
        })
    replaced = any(event['kind'] == 'generation_started' and event['generation_revision'] == update
                   for event in run['journal'])
    cancel_attempts = sum(event['kind'] == 'cancel_requested' and event['revision'] == update
                          for event in run['journal'])
    initial_committed_after_update = any(commit['generation_revision'] == run['initial_revision']
                                         and commit['after_update'] for commit in scored)
    return {
        'commit_scores': scored,
        'update_changed_oracle': before != after if update is not None else None,
        'false_replacement': replaced and before == after if update is not None else None,
        'missed_replacement': not replaced and before != after if update is not None else None,
        'update_cancel_attempts': cancel_attempts,
        'stale_units_dropped_by_fence': sum(event['kind'] == 'stale_unit_dropped' for event in run['journal']),
        'initial_committed_after_update': initial_committed_after_update if update is not None else None,
        'replacement_started': replaced,
        'replacement_committed': any(commit['generation_revision'] == update for commit in scored),
        'replacement_current_label_present': any(commit['generation_revision'] == update
                                                 and commit['current_label_present'] for commit in scored),
        'commits_with_only_outdated_label': sum(commit['only_outdated_label'] for commit in scored),
    }


async def execute(model):
    runs = []
    for case in CASES:
        for arm in ARMS:
            run = await run_case(model, case, arm)
            run['scores'] = score_run(run)
            runs.append(run)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from evaluation.local_qwen_adapter import LocalQwenAdapter
    from evaluation.provenance import capture

    provenance = capture([
        'evaluation/bounded_online_memory.py', 'evaluation/bounded_online_repair.py',
        'evaluation/local_qwen_adapter.py', 'evaluation/qwen_scheduler_smoke.py', 'runtime/continuous.py',
    ])
    model = LocalQwenAdapter(args.model_dir)
    report = {
        'scope': 'Six scripted cases crossed with three memory arms using one fixed decoder.',
        'capacity': CAPACITY, 'max_new_tokens': MAX_NEW_TOKENS, 'system_prompt': SYSTEM,
        'schedule': 'Whole-answer host delivery is held in the during-generation condition. '
                    'Worker overlap measures the inference job interval, including rendering and decoding. '
                    'Worker jobs drain on cancellation; timestamps include the scripted arrival conditions.',
        'scoring': 'Label presence uses literal matching. Mention of an old label can form part of a '
                   'withdrawal. Explicit withdrawal requires separate review of the committed text.',
        'checkpoint_files_sha256': model.file_hashes, 'provenance': provenance,
        'runs': asyncio.run(execute(model)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    if any(run['generation_errors'] for run in report['runs']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
