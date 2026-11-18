"""Replay source-scoped alias state against the unchanged v1 baselines."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

from evaluation.alias_online_compare import ANSWER_TOKENS, LEXICAL_LIMIT, SOURCES as V1_SOURCES, json_bytes, replay
from evaluation.alias_online_state import messages
from evaluation.alias_scoped_state import ScopedState, contains_mention, identifiers, normalize_text
from evaluation.continuous_event_protocol import load_streams, public_stream


FORMAT = 'pbslm.alias_online_predictions.v2'
ARMS = ('history', 'lexical', 'structured', 'scoped')
PURPOSES = ('maintenance', 'link', 'answer')
SOURCES = (*V1_SOURCES, 'evaluation/alias_online_compare_v2.py',
           'evaluation/alias_scoped_state.py', 'evaluation/score_alias_online_v2.py',
           'evaluation/provenance.py')
EXTRACT_PROMPT = (
    '输入是按到达顺序排列的事件JSON数组。只输出JSON操作数组，每项严格为'
    '[事件ID,操作,物件,值,来源]。'
    'set的物件是物件名、值是地点。clear的物件是物件名、值为已注销的旧地点或空。'
    'bind/rebind的物件是规范物件名、值是别名、来源是明确使用该称呼的班组或记录。'
    'revoke的物件是规范物件名、值是被撤销的别名、来源是明确的班组或记录。'
    'register的物件是新物件，值和来源为空，仅在原文明确该物件无存放地址时使用。'
    '地点事实、别名事实、来源范围是不同字段，不能互相替代。'
    '只根据对应事件原文输出。set只用于明确确认的实际存放关系。计划、假设、否认、'
    '未完成动作和无关语句不能成为set。地点或别名被明确撤销才输出clear或revoke。'
    '同一事件有无先后依据的冲突写[]。'
    '例：事件e1文字为“早班称工具箱A14为小扳手”时，输出'
    '[["e1","bind","工具箱A14","小扳手","早班"]]。'
    '例：事件e2文字为“工具箱A14已放入东侧柜”时，输出'
    '[["e2","set","工具箱A14","东侧柜",""]]。'
    '例：事件e3文字为“撤销东侧柜内工具箱A14的地点登记”时，输出'
    '[["e3","clear","工具箱A14","东侧柜",""]]。'
    '例：事件e4文字为“车间不再称J37为月牙轴”时，输出'
    '[["e4","revoke","J37","月牙轴","车间"]]。'
)
SCOPED_EXTRACT_TOKENS = 192
LOCATION_CUES = ('存放在', '暂置于', '暂放在', '暂存于', '放在', '放入', '放进', '放到',
                 '收在', '锁在', '夹在', '置于', '搁在', '摆在', '位于', '存于', '移入', '移到',
                 '搬到', '搬进', '迁到', '交到', '放回')
SPECULATIVE_CUES = ('如果', '要是', '假如', '提议', '计划', '打算', '拟把', '拟将',
                    '尚未', '还没', '没有', '未完成', '未移动', '没把')
INVALIDATION_CUES = ('撤销', '取消', '注销', '作废', '失效', '不再有效')
REVOKE_CUES = ('不再称', '不再用', '撤销称', '取消称', '注销别名', '撤销别名')


def maintenance_needed(text, memory):
    normalized = normalize_text(text)
    if any(cue in normalized for cue in LOCATION_CUES):
        return True
    has_entity_cue = bool(identifiers(normalized)) or any(
        contains_mention(normalized, name) for name in (*memory.entities, *memory.aliases)
    )
    has_memory_cue = any(cue in normalized for cue in (
        *INVALIDATION_CUES, *REVOKE_CUES, '简称', '称为', '称作',
        '叫作', '改称', '改为', '别名', '无存放地址', '没有地址',
    ))
    return has_memory_cue and (has_entity_cue or any(cue in normalized for cue in (
        '无存放地址', '没有地址',
    )))


def parse_operations(raw, observations):
    cleaned = re.sub(r'\A\x60\x60\x60(?:json)?\s*|\s*\x60\x60\x60\Z', '', raw.strip())
    try:
        rows = json.loads(cleaned)
    except (TypeError, ValueError):
        return {}, [{'row': None, 'reason': 'invalid JSON'}]
    if not isinstance(rows, list):
        return {}, [{'row': None, 'reason': 'not an array'}]
    sources = {row['event_id']: normalize_text(row['text']) for row in observations}
    accepted, rejected = defaultdict(list), []
    for row in rows:
        if (not isinstance(row, list) or len(row) != 5
                or any(not isinstance(value, str) for value in row)):
            rejected.append({'row': row, 'reason': 'invalid operation shape'})
            continue
        event_id, operation, entity, value, scope = (part.strip() for part in row)
        source = sources.get(event_id)
        if source is None:
            rejected.append({'row': row, 'reason': 'event ID is outside the pending prefix'})
            continue
        if operation not in {'register', 'set', 'clear', 'bind', 'rebind', 'revoke'}:
            rejected.append({'row': row, 'reason': 'unsupported operation'})
        elif not entity or any(not contains_mention(source, normalize_text(item))
                               for item in (entity, value, scope) if item):
            rejected.append({'row': row, 'reason': 'argument absent from current observation'})
        elif operation in {'bind', 'rebind'} and not scope and not re.search(
                r'各(?:个)?(?:班组|来源|部门|团队)通用|所有(?:班组|来源|部门|团队)(?:都)?(?:通用|适用)',
                source):
            rejected.append({'row': row, 'reason': 'alias binding requires an explicit source scope'})
        elif operation == 'set' and not _grounded_location(source, entity, value):
            rejected.append({'row': row, 'reason': 'set lacks a positive entity-location relation'})
        elif operation == 'clear' and not _grounded_clear(source, entity, value):
            rejected.append({'row': row, 'reason': 'clear lacks a target-linked invalidation cue'})
        elif operation == 'revoke' and not any(cue in source for cue in REVOKE_CUES):
            rejected.append({'row': row, 'reason': 'revoke lacks an explicit alias-revocation cue'})
        elif operation == 'register' and not any(cue in source for cue in ('无存放地址', '没有地址', '去向不明')):
            rejected.append({'row': row, 'reason': 'register lacks explicit missing-location evidence'})
        else:
            accepted[event_id].append((operation, entity, value, scope or None))

    grouped = defaultdict(list)
    for event_id, event_rows in accepted.items():
        for row in event_rows:
            if row[0] in {'set', 'clear'}:
                grouped[event_id, normalize_text(row[1])].append(row)
    for (event_id, surface), event_rows in grouped.items():
        if len({(row[0], normalize_text(row[2])) for row in event_rows}) <= 1:
            continue
        accepted[event_id] = [row for row in accepted[event_id]
                              if not (row[0] in {'set', 'clear'} and normalize_text(row[1]) == surface)]
        rejected.extend({'row': [event_id, *row],
                         'reason': 'contradictory same-observation location updates'}
                        for row in event_rows)
    return accepted, rejected


def _grounded_location(source, entity, value):
    if not value:
        return False
    for match in re.finditer(re.escape(normalize_text(value)), source):
        start = match.start()
        clause_start = max((source.rfind(mark, 0, start) for mark in ',。;!?'), default=-1) + 1
        clause_end_candidates = [position for mark in ',。;!?'
                                 if (position := source.find(mark, match.end())) >= 0]
        clause_end = min(clause_end_candidates, default=len(source))
        clause = source[clause_start:clause_end]
        prefix = source[max(clause_start, start - 12):start]
        if (contains_mention(clause, normalize_text(entity))
                and any(cue in prefix for cue in LOCATION_CUES)
                and not any(cue in clause for cue in SPECULATIVE_CUES)):
            return True
    return False


def _grounded_clear(source, entity, value):
    for clause in re.split(r'[,，;；。!?！？]', source):
        if (any(cue in clause for cue in INVALIDATION_CUES)
                and contains_mention(clause, normalize_text(entity))
                and (not value or contains_mention(clause, normalize_text(value)))):
            return True
    return False


def replay_scoped(generate, stream):
    started = time.perf_counter()
    memory = ScopedState()
    calls, predictions = [], []
    elapsed = defaultdict(float)
    cost = Counter()
    peak_bytes, observation_count = 0, 0
    event_id = None
    pending = []
    ingress_by_id = {}
    identity = {'arm': 'scoped', 'stream_id': stream['stream_id']}

    def maintain(prompt, observation_ids):
        begin = time.perf_counter()
        output, metrics = generate(prompt, max_new_tokens=SCOPED_EXTRACT_TOKENS)
        seconds = time.perf_counter() - begin
        elapsed['maintenance'] += seconds
        cost['maintenance'] += 1
        cost['maintenance', 'input'] += metrics['input_tokens']
        cost['maintenance', 'output'] += metrics['output_tokens']
        calls.append({'event_id': event_id, 'query_event_id': event_id, 'purpose': 'maintenance',
                      'observation_event_ids': observation_ids, 'messages': prompt,
                      'raw_output': output, 'generation': metrics, 'service_seconds': seconds})
        return output

    for event in stream['events']:
        event_id, text = event['event_id'], event['text']
        if event['kind'] == 'observation':
            observation_count += 1
            pending.append(event)
            ingress_by_id[event_id] = {
                'event_id': event_id, 'validated_operations': [], 'applied_operations': [],
                'rejected_parse': [], 'rejected_state': [], 'maintenance_skipped': True,
            }
        else:
            query_started = time.perf_counter()
            selected = [row for row in pending if maintenance_needed(row['text'], memory)]
            if selected:
                batch = [{'event_id': row['event_id'], 'text': row['text']} for row in selected]
                batch_ids = [row['event_id'] for row in selected]
                raw = maintain(messages(EXTRACT_PROMPT, json.dumps(batch, ensure_ascii=False)), batch_ids)
                operations, rejected_parse = parse_operations(raw, selected)
                rejected_by_id = defaultdict(list)
                for item in rejected_parse:
                    event_ref = item['row']
                    if isinstance(event_ref, list) and event_ref and event_ref[0] in batch_ids:
                        rejected_by_id[event_ref[0]].append(item)
                    else:
                        rejected_by_id[selected[-1]['event_id']].append(item)
                for row in selected:
                    ops = operations.get(row['event_id'], [])
                    aliases_in_event = {
                        normalize_text(value) for operation, _, value, _ in ops
                        if operation in {'bind', 'rebind'}
                    }
                    planned = []
                    for operation, surface, value, scope in ops:
                        can_register = (operation in {'bind', 'rebind'}
                                        or operation == 'set' and (scope is None or identifiers(surface)))
                        surface_key = normalize_text(surface)
                        if (can_register and surface_key not in memory.aliases
                                and not (operation == 'set' and surface_key in aliases_in_event)):
                            planned.append(('register', surface, '', None))
                        planned.append((operation, surface, value, scope))
                    applied, rejected_state = memory.apply(planned, row['event_id'])
                    trace = ingress_by_id[row['event_id']]
                    trace.update({
                        'validated_operations': [list(operation) for operation in planned],
                        'applied_operations': applied,
                        'rejected_parse': rejected_by_id[row['event_id']],
                        'rejected_state': rejected_state, 'maintenance_skipped': False,
                    })
            pending = []
            result = memory.lookup(text)
            evidence = result['evidence_event_ids']
            predictions.append({
                **identity, 'event_id': event_id, **result,
                'answer': result['readout'] if result['readout'] is not None else '不确定',
                'memory_evidence_event_ids': evidence, 'fallback': False,
                'available_observations': observation_count, 'messages': [],
                'query_seconds': time.perf_counter() - query_started,
            })
        peak_bytes = max(peak_bytes, memory.payload_bytes() + json_bytes(pending))

    total_seconds = time.perf_counter() - started
    model_seconds = sum(elapsed.values())
    totals = {
        **identity, 'calls': len(calls),
        'input_tokens': sum(cost[k, 'input'] for k in PURPOSES),
        'output_tokens': sum(cost[k, 'output'] for k in PURPOSES),
        'model_seconds': model_seconds, 'total_seconds': total_seconds,
        'host_seconds': total_seconds - model_seconds, 'maintenance_calls': cost['maintenance'],
        'linker_calls': cost['link'], 'answer_calls': cost['answer'],
        'fallback_queries': 0, 'peak_state_bytes': peak_bytes,
    }
    for purpose in PURPOSES:
        totals[f'{purpose}_input_tokens'] = cost[purpose, 'input']
        totals[f'{purpose}_output_tokens'] = cost[purpose, 'output']
        totals[f'{purpose}_model_seconds'] = elapsed[purpose]
    return predictions, totals, {**identity, 'ingress': list(ingress_by_id.values()), 'calls': calls}


def compare(generate, streams, arms):
    result = {'arms': list(arms), 'predictions': [], 'costs': [], 'ingress': []}
    for stream in streams:
        for arm in arms:
            if arm == 'scoped':
                predictions, totals, trace = replay_scoped(generate, stream)
            else:
                predictions, totals, trace = replay(
                    generate, stream, arm, record_query_seconds=True,
                )
            result['predictions'].extend(predictions)
            result['costs'].append(totals)
            result['ingress'].append(trace)
            print(f"{stream['stream_id']} {arm}: {totals['calls']} calls, "
                  f"{totals['input_tokens'] + totals['output_tokens']} tokens",
                  flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expect-fixture-sha256')
    parser.add_argument('--reverse-arms', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fixture_sha = hashlib.sha256(args.fixtures.read_bytes()).hexdigest()
    if args.expect_fixture_sha256 is not None and fixture_sha != args.expect_fixture_sha256:
        raise ValueError('fixture bytes do not match the frozen SHA-256')
    streams = [public_stream(stream) for stream in load_streams(args.fixtures)]
    if hashlib.sha256(args.fixtures.read_bytes()).hexdigest() != fixture_sha:
        raise ValueError('fixture changed while loading')

    from evaluation.local_qwen_adapter import LocalQwenAdapter
    from evaluation.provenance import capture
    from tokenizers import __version__ as tokenizers_version
    from transformers import __version__ as transformers_version

    provenance = capture(SOURCES)
    provenance['transformers'] = transformers_version
    provenance['tokenizers'] = tokenizers_version
    model = LocalQwenAdapter(args.model_dir)
    config = {
        'decode': 'BF16 greedy non-thinking', 'lexical_limit': LEXICAL_LIMIT,
        'answer_max_new_tokens': ANSWER_TOKENS, 'reverse_arms': args.reverse_arms,
        'scoped_extractor_max_new_tokens': SCOPED_EXTRACT_TOKENS,
        'scoped_maintenance': 'batch pending observations before each query',
        'scoped_answer': 'direct state readout',
        'fixture_name': args.fixtures.name,
    }
    arms = tuple(reversed(ARMS)) if args.reverse_arms else ARMS
    report = {
        'format': FORMAT, 'fixture_sha256': fixture_sha,
        'checkpoint_files_sha256': model.file_hashes, 'provenance': provenance,
        'config': config, **compare(model.generate, streams, arms),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
