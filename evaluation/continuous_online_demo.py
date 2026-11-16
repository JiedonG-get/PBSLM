"""Continuous Online event/output scenario and real fixed PBSLM query backend.

No training, feedback writes, parameter updates or language understanding claim.
Deterministic backend tests continuation text; real backend answers one token/query.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

from runtime.continuous import ContinuousRunner
from runtime.pbslm_backend import PBSLMQueryBackend
from runtime.model_session import model_fingerprint,state_hash
from evaluation.provenance import capture
from evaluation.text_facts import ENTITIES,TEMPLATES

ROOT=Path(__file__).resolve().parents[1]


class ScenarioBackend:
    def __init__(self):self.requests=[];self.release=asyncio.Event()

    async def generate(self,request):
        self.requests.append({'revision':request.revision,'inputs':request.inputs,
                              'committed_prefix':request.committed_prefix})
        if request.revision==1:
            yield '已收到：'
            await self.release.wait()
            yield '旧结论（不应提交）'
        else:
            yield '根据补充信息，'
            await asyncio.sleep(.001)
            yield '重新组织回答。'


async def scenario():
    backend=ScenarioBackend();runner=ContinuousRunner(backend)
    await runner.observe('第一条信息',should_respond=True)
    await runner.wait_event(lambda x:x['kind']=='output_committed')
    first_prefix=runner.output
    await runner.observe('补充信息改变了条件',should_respond=True)
    await runner.wait_event(lambda x:x['kind']=='generation_finished' and x['generation_revision']==2)
    output=runner.output
    await runner.observe('继续到达，但暂时只听',should_respond=False)
    await runner.close()
    stale_after=[item for item in runner.committed if item['revision']==1 and item['unit']!='已收到：']
    if stale_after or not output.startswith(first_prefix) or backend.requests[-1]['committed_prefix']!=first_prefix:
        raise AssertionError('scenario interruption/resumption failed')
    return {'backend':'deterministic scripted scenario, no neural reasoning',
            'output':output,'requests':backend.requests,'journal':runner.journal,
            'committed_prefix_preserved':True,'stale_suffix_committed':False,
            'incoming_input_during_active_generation':True}


async def real(checkpoint):
    backend=PBSLMQueryBackend(checkpoint,device='cpu',unit_delay=.02)
    before_model=model_fingerprint(backend.model)
    before_memory=state_hash(backend.model,memory=True)
    runner=ContinuousRunner(backend)
    inputs=[]
    try:
        # Await the first visible result to prove actual model output, then a burst.
        text=TEMPLATES[0].format(entity=ENTITIES[0])
        await runner.observe(text)
        await runner.wait_event(lambda x:x['kind']=='output_committed',timeout=15)
        for i in range(12):
            text=TEMPLATES[i%6].format(entity=ENTITIES[i%8])
            before=time.perf_counter()
            revision=await runner.observe(text,should_respond=i%4!=3)
            inputs.append({'revision':revision,'observe_started_perf':before,
                           'observe_returned_perf':time.perf_counter(),'text':text,
                           'should_respond':i%4!=3})
            await asyncio.sleep(.005)
        await runner.resume()
        final_revision=runner.revision
        await runner.wait_event(lambda x:x['kind']=='generation_finished' and x['generation_revision']==final_revision,timeout=15)
        await runner.close()
    finally:
        try:
            await runner.close()
        finally:
            # Always request worker shutdown even if draining asyncio tasks fails.
            # Running native inference is not preempted and may delay shutdown.
            await asyncio.to_thread(backend.shutdown)
    overlaps=[]
    for arrival in inputs:
        matching=[m['revision'] for m in backend.metrics if
                  m['inference_started_perf']<=arrival['observe_started_perf']<m['inference_finished_perf']]
        if matching:overlaps.append({'input_revision':arrival['revision'],'during_inference_revisions':matching})
    stale_commits=[]
    for event in runner.journal:
        if event['kind']=='output_committed' and event['generation_revision']!=event['revision']:
            stale_commits.append(event)
    if stale_commits:raise AssertionError('real backend stale commit')
    if before_model!=model_fingerprint(backend.model) or before_memory!=state_hash(backend.model,memory=True):
        raise AssertionError('fixed online runtime changed model or M')
    return {'backend':'real fixed PBSLMv2 CPU inference thread, one answer token per query',
            'model_and_memory_unchanged':True,
            'checkpoint':backend.artifact,'inputs':inputs,'inference_metrics':backend.metrics,
            'journal':runner.journal,'committed':runner.committed,
            'arrival_during_actual_compute':overlaps,
            'max_observe_call_seconds':max(i['observe_returned_perf']-i['observe_started_perf'] for i in inputs),
            'no_stale_commits':True,'model_context_limit':'only latest encoded question consumed; prefix/context received but not semantically processed',
            'gpu_preemption':False,'output_pacing_seconds':.02}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='experiments/text_runtime/models800/seed0.pt')
    p.add_argument('--output',default='experiments/continuous_online/demo.json');args=p.parse_args()
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    async def execute():return {'scenario':await scenario(),'real':await real(ROOT/args.checkpoint)}
    report=asyncio.run(execute())
    report['priority']='continuous Online running first; no Online Learning in this run'
    report['provenance']=capture(['runtime/continuous.py','runtime/pbslm_backend.py','evaluation/continuous_online_demo.py'])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'scenario_output':report['scenario']['output'],
                      'actual_inference_overlap_events':len(report['real']['arrival_during_actual_compute']),
                      'max_observe_call_seconds':report['real']['max_observe_call_seconds'],
                      'no_stale_commits':report['real']['no_stale_commits']},ensure_ascii=False))


if __name__=='__main__':main()
