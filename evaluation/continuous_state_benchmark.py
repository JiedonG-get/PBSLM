"""Fixed-reader continuous-runtime benchmark: transport, state and stale output separate.

Known entity/color templates; scripted should_respond, symbolic beliefs and renderer.
No training, no model M. Learned reader's accepted assertions are not verified truth.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from runtime.continuous import ContinuousRunner
from runtime.continuous_state import ContinuousStateBackend,ExactReferenceExtractor,LearnedReaderExtractor
from evaluation.text_facts import ENTITIES,COLORS
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


def sequence(seed,count,hard=False):
    rng=random.Random(seed+47000);rows=[]
    for i in range(count):
        e=rng.randrange(8);c=rng.randrange(8)
        kind=('assertion','correction','negated','uncertain')[i%4]
        forms={'assertion':f'{ENTITIES[e]}的颜色是{COLORS[c]}。',
               'correction':f'把{ENTITIES[e]}的颜色更新为{COLORS[c]}。',
               'negated':f'{ENTITIES[e]}不是{COLORS[c]}色。',
               'uncertain':f'不能确认{ENTITIES[e]}是不是{COLORS[c]}色。'}
        if hard and not kind in ('assertion','correction'):
            forms['negated']=f'别把{ENTITIES[e]}当成{COLORS[c]}色。'
            forms['uncertain']=f'传闻{ENTITIES[e]}为{COLORS[c]}色，但尚未证实。'
        rows.append({'event':i,'entity':e,'color':c,'kind':kind,'text':forms[kind],
                     'updates_truth':kind in ('assertion','correction')})
    return rows


def parameter_hash(extractor):
    if not hasattr(extractor,'model'):return None
    h=hashlib.sha256()
    for name,value in extractor.model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def percentile(values,q):
    ordered=sorted(values)
    return ordered[min(len(ordered)-1,int((len(ordered)-1)*q))] if ordered else None


async def evaluate(extractor,events):
    backend=ContinuousStateBackend(extractor,unit_delay=.02,state_capacity=16,snapshot_limit=128)
    runner=ContinuousRunner(backend);truth={};records=[];before=parameter_hash(extractor)
    latencies=[];services=[];stale_answers=0;opportunities=0
    try:
        for event in events:
            # Start output using currently known state, then deliver next actual input
            # after intro but before answer. No truth is passed to backend.
            if event['event']%8==0 and truth:
                e=sorted(truth)[0]
                q=await runner.observe(f'{ENTITIES[e]}的颜色是什么？')
                await runner.wait_event(lambda x:x['kind']=='output_committed' and x['generation_revision']==q)
                opportunities+=1
            else:q=None
            should_respond=event['event']%3==0
            start=time.perf_counter()
            revision=await runner.observe(event['text'],should_respond=should_respond)
            services.append(time.perf_counter()-start)
            state=await backend.processed(revision)
            latencies.append(time.perf_counter()-start)
            if event['updates_truth']:truth[event['entity']]=event['color']
            correct=sum(e in state and state[e]['color']==c for e,c in truth.items())
            extras=sum(e not in truth for e in state)
            expected={str(e):c for e,c in truth.items()}
            received={str(e):item['color'] for e,item in state.items()}
            extraction=next(x['extraction'] for x in reversed(backend.journal)
                            if x['kind']=='understanding_finished' and x['revision']==revision)
            records.append({'event':event,'input_revision':revision,'should_respond':should_respond,
                            'expected_state':expected,'actual_state':received,'known_correct':correct,
                            'known_total':len(truth),'extra_entities':extras,
                            'exact_state_match':expected==received,'extraction':extraction,
                            'observe_service_seconds':services[-1],'arrival_to_processed_seconds':latencies[-1]})
            if q is not None:
                # Revision fencing applies even if old work finishes after correction.
                if any(x['revision']==q and '颜色记录为' in x['unit'] for x in runner.committed):
                    stale_answers+=1
            if should_respond:
                await runner.wait_event(lambda x:x['kind']=='generation_finished' and x['generation_revision']==revision)
        await runner.observe('继续观察，不必回答。',should_respond=False)
        await backend.processed(runner.inputs[-1][0])
    finally:
        try:await runner.close()
        finally:await backend.aclose()
    if parameter_hash(extractor)!=before:raise AssertionError('reader parameters changed')
    stale_commits=[e for e in runner.journal if e['kind']=='output_committed' and e['revision']!=e['generation_revision']]
    if stale_commits:raise AssertionError('stale revision committed')
    negative=[r for r in records if not r['event']['updates_truth']]
    return {'events':records,'runtime_journal':runner.journal,'ingest_journal':backend.journal,
            'metrics':{'fact_events':len(records),'negative_inputs':len(negative),
                       'negative_accepted':sum(r['extraction'].get('accepted',False) for r in negative),
                       'mean_known_fact_accuracy':statistics.mean(r['known_correct']/r['known_total'] for r in records),
                       'exact_state_match_rate':statistics.mean(r['exact_state_match'] for r in records),
                       'p50_input_to_processed_seconds':percentile(latencies,.5),
                       'p95_input_to_processed_seconds':percentile(latencies,.95),
                       'max_observe_service_seconds':max(services),
                       'speaking_interruption_opportunities':opportunities,'old_assertions_after_interrupt':stale_answers,
                       'stale_revision_commits':len(stale_commits),'parameters_unchanged':True},
            'limits':['single known relation','symbolic belief state/rule speech','no PBSLM semantic generation',
                      'arrival timestamp means observe coroutine entry, not physical sensor time',
                      'sequentially wait for processed result; not high-rate overload stress']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--events',type=int,default=64)
    p.add_argument('--hard-negatives',action='store_true')
    p.add_argument('--output',default='experiments/continuous_online/state_benchmark64.json');args=p.parse_args()
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    if args.events<16:p.error('need >=16 events')
    async def execute():
        rows=[]
        for seed in args.seeds:
            seq=sequence(seed,args.events,args.hard_negatives)
            arms={'reference':await evaluate(ExactReferenceExtractor(),seq),
                  'learned_reader':await evaluate(LearnedReaderExtractor(ROOT/f'experiments/fact_reader/three_seed600/seed{seed}.pt'),seq)}
            rows.append({'seed':seed,'arms':arms})
            print(json.dumps({'seed':seed,'metrics':{name:a['metrics'] for name,a in arms.items()}}),flush=True)
        return rows
    report={'config':vars(args),'runs':asyncio.run(execute()),
            'provenance':capture(['runtime/continuous.py','runtime/continuous_state.py','runtime/fact_reader.py',
                                  'evaluation/continuous_state_benchmark.py']),
            'scope':'fixed-weight ongoing ingest/state/interrupt reference task, not Online Learning'}
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
