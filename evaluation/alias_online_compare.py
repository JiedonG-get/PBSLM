"""Paired causal replay of Chinese location queries using one frozen Qwen backend."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

from evaluation.alias_online_state import AliasState, EXTRACT_PROMPT, EXTRACT_TOKENS, messages, parse_updates
from evaluation.alias_retrieval import rank_context
from evaluation.continuous_event_protocol import load_streams, public_stream


ARMS = ('history', 'lexical', 'structured', 'structured_fallback')
LEXICAL_LIMIT = 8
ANSWER_TOKENS = 24
ANSWER_PROMPT = (
    '回答物件的存放地点。只写地点名称（如“器具室”），不要写物件名或解释。'
    '若输入的“当前确认地点”字段给出了地点，原样输出该值；该字段为 null 则回答“不确定”。'
    '若给的是按时间排列的记录，先用编号、已登记的别名或明确的上问指代确定物件，再找最新有效地点。'
    '计划、传闻、纯否认和未完成的动作不更改地点。明确作废的记录失效。'
    '若物件或地点无法确定，只写“不确定”。'
)
SOURCES = (
    'evaluation/alias_online_compare.py', 'evaluation/alias_online_state.py',
    'evaluation/alias_retrieval.py', 'evaluation/continuous_event_protocol.py',
    'evaluation/local_qwen_adapter.py',
)


def json_bytes(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def answer_messages(question, previous_question, *, evidence=None, entity=None, readout=None):
    if evidence is not None:
        facts = [row['text'] for row in evidence]
        content = {'已到达的记录（从旧到新）': facts}
    else:
        content = {'识别的物件': entity, '当前确认地点': readout}
    content['上个问题'] = previous_question
    content['当前问题'] = question
    return messages(ANSWER_PROMPT, json.dumps(content, ensure_ascii=False))


def replay(generate, stream, arm, *, record_query_seconds=False):
    started = time.perf_counter()
    memory = AliasState() if arm.startswith('structured') else None
    keep_history = arm != 'structured'
    history, predictions, calls, ingress = [], [], [], []
    previous_question = None
    cost = Counter()
    elapsed = defaultdict(float)
    peak_bytes = 0
    event_id = None

    def call(purpose, prompt, budget):
        begin = time.perf_counter()
        output, metrics = generate(prompt, max_new_tokens=budget)
        seconds = time.perf_counter() - begin
        elapsed[purpose] += seconds
        cost[purpose] += 1
        cost[purpose, 'input'] += metrics['input_tokens']
        cost[purpose, 'output'] += metrics['output_tokens']
        calls.append({'event_id': event_id, 'purpose': purpose, 'messages': prompt,
                      'raw_output': output, 'generation': metrics, 'service_seconds': seconds})
        return output

    for event in stream['events']:
        event_id, text = event['event_id'], event['text']
        if event['kind'] == 'observation':
            if keep_history:
                history.append(event)
            if memory is not None:
                output = call('maintenance', messages(EXTRACT_PROMPT, text), EXTRACT_TOKENS)
                rows, rejected_parse = parse_updates(output, text)
                applied, rejected_state = memory.apply(rows, event_id)
                ingress.append({'event_id': event_id, 'validated_operations': rows, 'applied_operations': applied,
                                'rejected_parse': rejected_parse, 'rejected_state': rejected_state})
        else:
            query_started = time.perf_counter() if record_query_seconds else None
            entity = readout = method = None
            evidence = []
            fallback = False
            if memory is not None:
                entity, readout, evidence, method = memory.lookup(text, call)
                fallback = arm == 'structured_fallback' and readout is None
            if memory is None or fallback:
                selected = rank_context(history, text, limit=None if arm == 'history' else LEXICAL_LIMIT)
                answer_prompt = answer_messages(text, previous_question, evidence=selected)
                answer_evidence = [row['event_id'] for row in selected]
            else:
                answer_prompt = answer_messages(text, previous_question, entity=entity, readout=readout)
                answer_evidence = evidence
            answer = call('answer', answer_prompt, ANSWER_TOKENS)
            cost['fallback_queries'] += fallback
            prediction = {'arm': arm, 'stream_id': stream['stream_id'], 'event_id': event_id,
                          'answer': answer, 'entity': entity, 'readout': readout, 'link_method': method,
                          'evidence_event_ids': answer_evidence, 'memory_evidence_event_ids': evidence,
                          'fallback': fallback, 'available_observations': len(history) if keep_history else None,
                          'messages': answer_prompt}
            if record_query_seconds:
                prediction['query_seconds'] = time.perf_counter() - query_started
            predictions.append(prediction)
            previous_question = text
        history_bytes = json_bytes(history) if keep_history else 0
        state_bytes = memory.payload_bytes() if memory is not None else 0
        peak_bytes = max(peak_bytes, history_bytes + state_bytes + json_bytes(previous_question))

    total_seconds = time.perf_counter() - started
    model_seconds = sum(elapsed.values())
    purposes = ('maintenance', 'link', 'answer')
    totals = {'arm': arm, 'stream_id': stream['stream_id'],
              'calls': len(calls), 'input_tokens': sum(cost[k, 'input'] for k in purposes),
              'output_tokens': sum(cost[k, 'output'] for k in purposes),
              'model_seconds': model_seconds, 'total_seconds': total_seconds,
              'host_seconds': total_seconds - model_seconds, 'maintenance_calls': cost['maintenance'],
              'maintenance_input_tokens': cost['maintenance', 'input'],
              'maintenance_output_tokens': cost['maintenance', 'output'],
              'maintenance_model_seconds': elapsed['maintenance'],
              'linker_calls': cost['link'], 'answer_calls': cost['answer'],
              'fallback_queries': cost['fallback_queries'], 'peak_state_bytes': peak_bytes}
    trace = {'arm': arm, 'stream_id': stream['stream_id'], 'ingress': ingress, 'calls': calls}
    return predictions, totals, trace


def compare(generate, streams, arms):
    result = {'arms': list(arms), 'predictions': [], 'costs': [], 'ingress': []}
    for stream in streams:
        for arm in arms:
            predictions, costs, trace = replay(generate, stream, arm)
            result['predictions'].extend(predictions)
            result['costs'].append(costs)
            result['ingress'].append(trace)
        print(stream['stream_id'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reverse-arms', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fixture_sha = hashlib.sha256(args.fixtures.read_bytes()).hexdigest()
    streams = [public_stream(stream) for stream in load_streams(args.fixtures)]
    if hashlib.sha256(args.fixtures.read_bytes()).hexdigest() != fixture_sha:
        raise ValueError('fixture changed while loading')

    from evaluation.local_qwen_adapter import LocalQwenAdapter
    from evaluation.provenance import capture
    from transformers import __version__ as transformers_version

    provenance = capture(SOURCES)
    provenance['transformers'] = transformers_version
    model = LocalQwenAdapter(args.model_dir)
    config = {'decode': 'BF16 greedy non-thinking', 'lexical_limit': LEXICAL_LIMIT,
              'answer_max_new_tokens': ANSWER_TOKENS, 'reverse_arms': args.reverse_arms,
              'fixture_name': args.fixtures.name}
    arms = tuple(reversed(ARMS)) if args.reverse_arms else ARMS
    report = {'format': 'pbslm.alias_online_predictions.v1', 'fixture_sha256': fixture_sha,
              'checkpoint_files_sha256': model.file_hashes, 'provenance': provenance, 'config': config,
              **compare(model.generate, streams, arms)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
