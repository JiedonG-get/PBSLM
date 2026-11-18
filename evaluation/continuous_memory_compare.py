"""Run paired query snapshots; scoring is a separate offline step."""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import unicodedata

from evaluation.continuous_event_protocol import load_streams, public_stream


ARMS = ('history', 'recent', 'retrieval', 'bounded_summary')
QUERY_INSTRUCTION = (
    '根据给出的观察回答当前问题。以已确认的最新信息为准，未经证实的猜测不能推翻已确认的事实。'
    '只输出简短答案，不加解释。没有足够证据时只输出“不确定”。'
)
UPDATE_INSTRUCTION = (
    '把上一份笔记和新观察合并成用于之后问答的简短事实笔记。'
    '用简短分句保留各对象已确认的当前事实与明确更正；保留原有仍有效的事实。'
    '未经证实的传闻不能推翻已确认事实。只输出更新后的笔记。'
)


def bigrams(text):
    letters = ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())
    return {letters[i:i + 2] for i in range(len(letters) - 1)}


def run_arm(generate, arm, stream, *, window, state_chars, query_tokens, update_tokens):
    evidence = deque(maxlen=window if arm == 'recent' else None)
    state = ''
    rows = []
    calls = []
    maintenance = []

    def call(event, purpose, instruction, content, budget):
        messages = [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': content}]
        started = time.perf_counter()
        answer, metrics = generate(messages, max_new_tokens=budget)
        calls.append({'arm': arm, 'stream_id': stream['stream_id'], 'event_id': event['event_id'],
                      'purpose': purpose, 'messages': messages, 'raw_output': answer,
                      'input_tokens': metrics['input_tokens'], 'output_tokens': metrics['output_tokens'],
                      'finished_by_eos': metrics['finished_by_eos'], 'max_new_tokens': budget,
                      'service_seconds': time.perf_counter() - started})
        return answer.strip()

    for event in stream['events']:
        text = event['text']
        if event['kind'] == 'observation':
            if arm == 'bounded_summary':
                candidate = call(event, 'update', UPDATE_INSTRUCTION,
                                 f'旧笔记\n{state}\n\n新观察\n{text}', update_tokens)
                state = candidate[:state_chars] if candidate else state
                maintenance.append({'event_id': event['event_id'], 'state': state,
                                    'character_limit_applied': len(candidate) > state_chars,
                                    'empty_update_kept_old_state': not candidate})
            else:
                evidence.append((event['event_id'], text, bigrams(text) if arm == 'retrieval' else None))
            continue

        selected = list(evidence)
        if arm == 'retrieval':
            query = bigrams(text)
            ranks = sorted(range(len(selected)), key=lambda i: (len(query & selected[i][2]), i), reverse=True)
            selected = [selected[i] for i in sorted(ranks[:window])]
        context = state if arm == 'bounded_summary' else '\n'.join(item[1] for item in selected)
        answer = call(event, 'query', QUERY_INSTRUCTION, f'观察资料\n{context}\n\n问题\n{text}', query_tokens)
        rows.append({'arm': arm, 'stream_id': stream['stream_id'], 'event_id': event['event_id'],
                     'answer': answer, 'evidence_event_ids': [item[0] for item in selected],
                     'evidence_chars': len(context), 'stored_evidence_rows': len(evidence),
                     'retrieval_rows_scanned': len(evidence) if arm == 'retrieval' else 0,
                     'scheduled_delivery_ms': event['deliver_after_ms']})

    return rows, calls, maintenance


def compare(generate, streams, **budgets):
    predictions = []
    calls = []
    state_updates = []
    for stream in streams:
        for arm in ARMS:
            arm_predictions, arm_calls, maintenance = run_arm(generate, arm, stream, **budgets)
            predictions.extend(arm_predictions)
            calls.extend(arm_calls)
            if maintenance:
                state_updates.append({'stream_id': stream['stream_id'], 'updates': maintenance})
    return {'arms': list(ARMS), 'predictions': predictions, 'calls': calls, 'state_updates': state_updates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=int, default=3)
    parser.add_argument('--state-chars', type=int, default=128)
    parser.add_argument('--query-tokens', type=int, default=24)
    parser.add_argument('--update-tokens', type=int, default=96)
    args = parser.parse_args()
    if min(args.window, args.state_chars, args.query_tokens, args.update_tokens) < 1:
        parser.error('budgets must be positive')
    if max(args.query_tokens, args.update_tokens) > 256:
        parser.error('generation budget cannot exceed 256 tokens')
    if args.output.exists():
        raise FileExistsError(args.output)

    from evaluation.local_qwen_adapter import LocalQwenAdapter
    from evaluation.provenance import capture

    fixture_sha = hashlib.sha256(args.fixtures.read_bytes()).hexdigest()
    streams = [public_stream(fixture) for fixture in load_streams(args.fixtures)]
    if hashlib.sha256(args.fixtures.read_bytes()).hexdigest() != fixture_sha:
        raise ValueError('fixture changed while loading')
    model = LocalQwenAdapter(args.model_dir, device=args.device)
    budgets = {key: getattr(args, key) for key in ('window', 'state_chars', 'query_tokens', 'update_tokens')}
    report = compare(model.generate, streams, **budgets)
    report.update({'fixture_sha256': fixture_sha, 'checkpoint_files_sha256': model.file_hashes,
                   'config': {**budgets, 'device': args.device, 'decoding': 'greedy, non-thinking'},
                   'scope': 'sequential synthetic query snapshots; no timed ingress, interruption or semantic continuation',
                   'provenance': capture(['evaluation/continuous_memory_compare.py',
                                          'evaluation/continuous_event_protocol.py',
                                          'evaluation/local_qwen_adapter.py'])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(f'{len(streams)} streams, {len(report["predictions"])} predictions, {len(report["calls"])} model calls')


if __name__ == '__main__':
    main()
