"""Replay the frozen bounded-memory streams with one frozen language backend."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from evaluation.continuous_event_protocol import load_streams, public_stream


FIXTURE_SHA256 = '98234923f06addd910af3f46b6db452c74c1b3a89b9f462faf535d1da7ccdbb7'
CAPACITY = 16
QUERY_TOKENS = 24
INSTRUCTION = (
    '用记忆中给出的存放位置回答问题。只输出地点名称，不能添加说明或标点。'
    '如果没有已确认的存放位置，只输出“不确定”。'
)


def query_messages(question, readout):
    evidence = f'已确认的存放位置：{readout}' if readout is not None else '没有已确认的存放位置'
    return [{'role': 'system', 'content': INSTRUCTION},
            {'role': 'user', 'content': f'记忆：{evidence}。\n问题：{question}'}]


def replay(generate, stream, arm, memory):
    started = time.perf_counter()
    rows = []
    trace = []
    prompt_tokens = output_tokens = 0
    service_seconds = 0.0
    peak_payload = memory.metrics()['persistent_payload_bytes']
    for event in stream['events']:
        event_id, message = event['event_id'], event['text']
        if event['kind'] == 'observation':
            written = memory.observe(message)
            footprint = memory.metrics()['persistent_payload_bytes']
            peak_payload = max(peak_payload, footprint)
            trace.append({'event_id': event_id, 'written': written, 'persistent_payload_bytes': footprint})
            continue

        readout = memory.answer(message)
        messages = query_messages(message, readout)
        call_started = time.perf_counter()
        answer, metrics = generate(messages, max_new_tokens=QUERY_TOKENS)
        elapsed = time.perf_counter() - call_started
        prompt_tokens += metrics['input_tokens']
        output_tokens += metrics['output_tokens']
        service_seconds += elapsed
        rows.append({'arm': arm, 'stream_id': stream['stream_id'], 'event_id': event_id,
                     'readout': readout, 'answer': answer, 'messages': messages,
                     'generation': metrics, 'service_seconds': elapsed})

    totals = {'arm': arm, 'stream_id': stream['stream_id'], 'memory': memory.metrics(),
              'peak_persistent_payload_bytes': peak_payload, 'calls': len(rows),
              'input_tokens': prompt_tokens, 'output_tokens': output_tokens,
              'model_service_seconds': service_seconds, 'wall_seconds': time.perf_counter() - started}
    return rows, totals, trace


def compare(generate, streams, arms, make_memory):
    predictions = []
    costs = []
    writes = []
    for stream in streams:
        for arm in arms:
            rows, totals, trace = replay(generate, stream, arm, make_memory(arm, capacity=CAPACITY))
            predictions.extend(rows)
            costs.append(totals)
            writes.append({'arm': arm, 'stream_id': stream['stream_id'], 'observations': trace})
    return {'arms': list(arms), 'predictions': predictions, 'costs': costs, 'observations': writes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=Path('evaluation/fixtures/bounded_online_v1.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--reverse-arms', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fixture_hash = hashlib.sha256(args.fixtures.read_bytes()).hexdigest()
    if fixture_hash != FIXTURE_SHA256:
        raise ValueError('input does not match the frozen benchmark')
    streams = [public_stream(stream) for stream in load_streams(args.fixtures)]
    if hashlib.sha256(args.fixtures.read_bytes()).hexdigest() != fixture_hash:
        raise ValueError('fixture changed while loading')

    from evaluation.bounded_online_memory import ARMS, make_memory
    from evaluation.local_qwen_adapter import LocalQwenAdapter
    from evaluation.provenance import capture
    from transformers import __version__ as transformers_version

    sources = ('evaluation/bounded_online_compare.py', 'evaluation/bounded_online_memory.py',
               'evaluation/fixtures/bounded_online_delta_seed0.json', 'models/content_fast_state.py',
               'models/dynamic_content_slots.py', 'evaluation/local_qwen_adapter.py',
               'evaluation/continuous_event_protocol.py')
    provenance = capture(sources)
    provenance['transformers'] = transformers_version
    model = LocalQwenAdapter(args.model_dir, device=args.device)
    arms = tuple(reversed(ARMS)) if args.reverse_arms else ARMS
    report = compare(model.generate, streams, arms, make_memory)
    report.update({'format': 'pbslm.bounded_online_predictions.v1',
                   'fixture_sha256': fixture_hash, 'checkpoint_files_sha256': model.file_hashes,
                   'config': {'capacity': CAPACITY, 'max_new_tokens': QUERY_TOKENS, 'device': args.device,
                              'decoding': 'BF16 greedy, non-thinking', 'reverse_arms': args.reverse_arms},
                   'provenance': provenance})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(f'{len(streams)} streams and {len(report["predictions"])} predictions')


if __name__ == '__main__':
    main()
