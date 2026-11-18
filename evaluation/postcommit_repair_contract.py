"""Test post-commit correction with natural, structured, and validated delivery."""

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import time
import unicodedata

from evaluation.bounded_online_memory import DEFAULT_LABELS, make_memory, parse_query
from evaluation.bounded_online_repair import MAX_NEW_TOKENS, SYSTEM, WAIT_SECONDS
from evaluation.fixtures.postcommit_repair_contract_v1 import DEVELOPMENT, FORMAL
from runtime.continuous import ContinuousRunner


ARMS = ('direct_prefix', 'structured_prompt', 'guarded_commit', 'deterministic_template')
STRUCTURED_SYSTEM = (
    '你根据已给出的两个字段修订样件的存放位置。只输出一个完整句子。'
    '如果已发地点字段包含地点，严格照此格式输出：撤回先前的地点说法（旧地点），更正为新地点。'
    '如果已发地点字段为不确定，严格照此格式输出：现已确认存放位置是新地点。'
    '把旧地点和新地点替换成字段里的原文，不添加其他文字。'
)
UNKNOWN = '不确定'
WITHDRAW_PREFIX = '撤回先前的地点说法'
UPDATE_PREFIX = '现已确认存放位置是'


def normalize(text):
    body = unicodedata.normalize('NFKC', text).strip()
    if body.endswith(('。', '.')):
        return body[:-1].rstrip()
    return body


def template(old, new):
    if old is None:
        return f'{UPDATE_PREFIX}{new}。'
    return f'{WITHDRAW_PREFIX}（{old}），更正为{new}。'


CLAIMS = {UNKNOWN: ('uncertain', None)}
for new in DEFAULT_LABELS:
    CLAIMS[new] = ('location', new)
    for old in (None, *DEFAULT_LABELS):
        if old != new:
            CLAIMS[normalize(template(old, new))] = ('location', new)


def resolution_or_repair(sentence):
    return CLAIMS.get(normalize(sentence), ('unrecognized', None))


@dataclass(frozen=True)
class Snapshot:
    question: str
    value: str | None
    initial: bool
    previous_status: str
    previous_value: str | None
    previous_count: int

    @property
    def eligible(self):
        return (not self.initial and self.value is not None
                and self.previous_status in {'location', 'uncertain'}
                and self.previous_value != self.value)


def render_messages(request, snapshot, arm):
    if snapshot.initial or arm == 'direct_prefix':
        messages = [{'role': 'system', 'content': SYSTEM}]
        if request.committed_prefix:
            messages.extend((
                {'role': 'user', 'content': snapshot.question},
                {'role': 'assistant', 'content': request.committed_prefix},
            ))
        value = snapshot.value if snapshot.value is not None else '暂无已确认的地点'
        messages.append({'role': 'user', 'content': f'{snapshot.question}\n当前资料：{value}。'})
        return messages

    previous = snapshot.previous_value or UNKNOWN
    if snapshot.previous_status not in {'location', 'uncertain'}:
        previous = '无法识别已发地点'
    return [
        {'role': 'system', 'content': STRUCTURED_SYSTEM},
        {'role': 'user', 'content': f'{snapshot.question}\n已发地点：{previous}\n'
                                    f'当前确认地点：{snapshot.value or UNKNOWN}'},
    ]


class DeliveryBackend:
    def __init__(self, backend, snapshots, arm, is_current):
        self.backend = backend
        self.snapshots = snapshots
        self.arm = arm
        self.is_current = is_current
        self.decisions = []

    async def generate(self, request):
        started = time.perf_counter()
        snapshot = self.snapshots[request.revision]
        controlled = not snapshot.initial and self.arm in {'guarded_commit', 'deterministic_template'}
        candidate = None
        if self.arm != 'deterministic_template' or snapshot.initial:
            if not controlled or snapshot.eligible:
                async for answer in self.backend.generate(request):
                    candidate = answer

        if not controlled:
            chosen, decision = candidate, 'model'
        elif not snapshot.eligible:
            chosen, decision = None, 'no_eligible_claim'
        elif not self.is_current(snapshot):
            chosen, decision = None, 'snapshot_no_longer_current'
        else:
            expected = template(snapshot.previous_value, snapshot.value)
            if self.arm == 'deterministic_template':
                chosen, decision = expected, 'template'
            elif candidate is not None and normalize(candidate) == normalize(expected):
                chosen, decision = candidate, 'verified_model'
            else:
                chosen, decision = expected, 'template_fallback'
        self.decisions.append({
            'revision': request.revision, 'eligible': snapshot.eligible,
            'reason': decision, 'model_candidate': candidate, 'offered_unit': chosen,
            'decision_seconds': time.perf_counter() - started,
        })
        if chosen:
            yield chosen


async def run_case(model, events, arm, *, backend_factory=None):
    if arm not in ARMS:
        raise ValueError(f'unknown repair arm {arm!r}')
    if backend_factory is None:
        from evaluation.qwen_scheduler_smoke import LocalQwenBackend

        backend_factory = LocalQwenBackend
    question = None
    value = None
    seen = {}
    snapshots = {}
    trace = []
    memory = make_memory('exact_unbounded')

    def render(request):
        return render_messages(request, snapshots[request.revision], arm)

    def is_current(snapshot):
        return (snapshot.question == question and snapshot.value == value
                and snapshot.previous_count == len(runner.committed))

    backend = backend_factory(model, MAX_NEW_TOKENS, render_messages=render)
    delivery = DeliveryBackend(backend, snapshots, arm, is_current)
    runner = ContinuousRunner(delivery, history_limit=1)
    try:
        for event in events:
            started = time.perf_counter()
            previous_value = value
            previous_count = len(runner.committed)
            previous_unit = runner.committed[-1]['unit'] if previous_count else None
            prior, spoken_value = resolution_or_repair(previous_unit) if previous_unit else ('absent', None)
            if event.identity in seen:
                if event.text != seen[event.identity]:
                    raise ValueError('an event identity was redelivered with different text')
                revision, route = None, 'duplicate_delivery'
            else:
                seen[event.identity] = event.text
                revision = runner.revision + 1
                is_query = event.text.startswith('请问')
                if is_query:
                    parse_query(event.text)
                    question = event.text
                    value = memory.answer(question)
                elif memory.observe(event.text) and question is not None:
                    value = memory.peek(question)
                if is_query or value != previous_value:
                    route = 'query' if is_query else 'update'
                    snapshots[revision] = Snapshot(question, value, is_query, prior, spoken_value, previous_count)
                    await runner.observe(event.text)
                else:
                    route = 'background'
                    await runner.observe_background(event.text, relevance_assertion='unmodified current evidence')
            routed = time.perf_counter()
            if route in {'query', 'update'}:
                await runner.wait_event(
                    lambda item: item['kind'] in {'generation_finished', 'generation_failed'}
                    and item['generation_revision'] == revision, timeout=WAIT_SECONDS,
                )
            trace.append({
                'event_id': event.identity, 'text': event.text, 'revision': revision, 'route': route,
                'evidence_before': previous_value, 'evidence_after': value,
                'previous_claim_status': prior, 'previous_claim_value': spoken_value,
                'prior_commit_count': previous_count, 'prior_committed_unit': previous_unit,
                'received_perf': started, 'routing_seconds': routed - started,
                'settled_seconds': time.perf_counter() - started,
            })
    finally:
        try:
            await runner.close(timeout=WAIT_SECONDS)
        finally:
            await backend.close()

    return {
        'arm': arm, 'trace': trace, 'committed': runner.committed, 'journal': runner.journal,
        'worker_jobs': backend.metrics, 'delivery_decisions': delivery.decisions,
    }


def score(case, run):
    if len(case.expectations) != len(run['trace']):
        raise ValueError('expectations and trace length differ')
    scored = []
    for (kind, old, new), step in zip(case.expectations, run['trace'], strict=True):
        units = [item['unit'] for item in run['committed'] if item['revision'] == step['revision']]
        decision = next((item for item in run['delivery_decisions'] if item['revision'] == step['revision']), None)
        exact_answer = resolution_or_repair(units[0]) if len(units) == 1 else None
        recognized_claim = step['previous_claim_status'] in {'location', 'uncertain'}
        if kind == 'silent':
            ok = not units and decision is None
        elif kind == 'withdraw_and_replace':
            ok = len(units) == 1 and normalize(units[0]) == normalize(template(old, new))
        elif kind == 'resolve_unknown':
            ok = len(units) == 1 and normalize(units[0]) in {new, normalize(template(None, new))}
        else:
            expected_answer = ('location', new) if new is not None else ('uncertain', None)
            ok = exact_answer == expected_answer
        scored.append({
            'event_id': step['event_id'], 'revision': step['revision'], 'expected_kind': kind,
            'gold_old': old, 'gold_new': new, 'committed_units': units, 'strict_pass': ok,
            'correct_readout': step['evidence_after'] == new if kind != 'silent' else None,
            'new_value_mentioned': (any(new in unit for unit in units) if new is not None else None),
            'prior_claim_recognized': recognized_claim,
            'prior_claim_matches_gold': (step['previous_claim_value'] == old and recognized_claim
                                         if kind in {'withdraw_and_replace', 'resolve_unknown'} else None),
            'delivery_reason': decision['reason'] if decision else None,
        })
    return scored


async def execute(model, cases, arms):
    runs = []
    for case in cases:
        for arm in arms:
            run = await run_case(model, case.events, arm)
            run['case'] = case.identity
            runs.append(run)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('development', 'formal'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reverse-arms', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from evaluation.local_qwen_adapter import LocalQwenAdapter
    from evaluation.provenance import capture

    provenance = capture([
        'evaluation/bounded_online_memory.py', 'evaluation/bounded_online_repair.py',
        'evaluation/fixtures/postcommit_repair_contract_v1.py', 'evaluation/local_qwen_adapter.py',
        'evaluation/postcommit_repair_contract.py', 'evaluation/qwen_scheduler_smoke.py',
        'evaluation/score_postcommit_repair_contract.py', 'runtime/continuous.py',
    ])
    model = LocalQwenAdapter(args.model_dir)
    cases = DEVELOPMENT if args.split == 'development' else FORMAL
    arms = ARMS[::-1] if args.reverse_arms else ARMS
    report = {
        'scope': 'Post-commit whole-answer repair with explicitly structured Chinese notices.',
        'split': args.split, 'arms': arms, 'max_new_tokens': MAX_NEW_TOKENS,
        'checkpoint_files_sha256': model.file_hashes, 'provenance': provenance,
        'runs': asyncio.run(execute(model, cases, arms)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    if any(item['kind'] == 'generation_failed' for run in report['runs'] for item in run['journal']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
