"""Saved learned content-fast state in actual asynchronous ingress/output loop.

Structured random key/value events, not natural-language comprehension or original
PBSLM integration. Gate makes interruption reproducible. No neural accuracy assert;
record wrong/unknown results honestly. No raw-history input in model query.
"""
import asyncio
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from runtime.fast_state_online import load_fast_model,FastStateOnlineBackend
from runtime.continuous import ContinuousRunner
from evaluation.continuous_key_control import state_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


def encode(op,key,value=None):
    row={'op':op,'key':key.tolist()}
    if op=='write':row['value']=value
    return json.dumps(row)


async def scenario(seed):
    model,artifact=load_fast_model(ROOT/f'experiments/recurrent_state/content_fast600/delta_seed{seed}.pt')
    frozen=state_hash(model);gate=asyncio.Event()
    backend=FastStateOnlineBackend(model,output_gate=gate);runner=ContinuousRunner(backend)
    gen=torch.Generator().manual_seed(seed+201000)
    keys=F.normalize(torch.randn(8,16,generator=gen),dim=-1);labels=torch.randint(8,(8,),generator=gen)
    services=[];answers={}
    try:
        for index in range(8):
            start=time.perf_counter()
            revision=await runner.observe(encode('write',keys[index],int(labels[index])),should_respond=False)
            services.append(time.perf_counter()-start)
            await backend.processed(revision)
        if runner.output:raise AssertionError('passive writes spoke')
        old_query=await runner.observe(encode('query',keys[0]))
        await runner.wait_event(lambda e:e['kind']=='output_committed' and e['generation_revision']==old_query)
        old=await backend.processed(old_query)
        prefix=runner.output
        revised=(int(labels[0])+1)%8
        correction=await runner.observe(encode('write',keys[0],revised),should_respond=False)
        corrected_state=await backend.processed(correction)
        query=await runner.observe(encode('query',keys[0]))
        await runner.wait_event(lambda e:e['kind']=='output_committed' and e['generation_revision']==query)
        gate.set()
        await runner.wait_event(lambda e:e['kind']=='generation_finished' and e['generation_revision']==query)
        latest=await backend.processed(query)
        if latest['state_before_sha256']!=corrected_state['state_after_sha256']:raise AssertionError('query not using corrected state')
        if any(e['revision']==old_query and e['unit']!='query_started;' for e in runner.committed):raise AssertionError('old result leaked')
        if not runner.output.startswith(prefix):raise AssertionError('committed prefix lost')
        answers={'before':{'target':int(labels[0]),'prediction':old['prediction'],'correct':old['prediction']==int(labels[0])},
                 'after':{'target':revised,'prediction':latest['prediction'],'correct':latest['prediction']==revised}}
        # Query untouched entity and a fresh unknown after correction; no output requested.
        for name,key,target in [('cold',keys[1],int(labels[1])),('unknown',F.normalize(torch.randn(16,generator=gen),dim=-1),8)]:
            r=await runner.observe(encode('query',key),should_respond=False)
            result=await backend.processed(r)
            answers[name]={'target':target,'prediction':result['prediction'],'correct':result['prediction']==target}
    finally:
        try:await runner.close()
        finally:await backend.aclose()
    if state_hash(model)!=frozen:raise AssertionError('parameters changed')
    return {'seed':seed,'checkpoint':artifact,'frozen_weights_sha256':frozen,'weights_unchanged':True,
            'answers':answers,'runtime_journal':runner.journal,'backend_journal':backend.journal,
            'max_passive_observe_service_seconds':max(services),'no_old_answer_after_correction':True,
            'prefix_preserved':True,'state_floats':128,'retained_result_limit':16,
            'model_query_inputs':'current key and fast-state matrix only',
            'warning':'runtime/evaluator logs contain event text; those logs are not read by neural state engine'}


def main():
    out=ROOT/'experiments/recurrent_state/content_fast600/online_adapter.json'
    if out.exists():raise FileExistsError(out)
    async def execute():return [await scenario(seed) for seed in (0,1,2)]
    report={'scope':'real learned fast-state asynchronous structured-event adapter, NOT original PBSLM model or language output',
            'runs':asyncio.run(execute()),'provenance':capture(['runtime/fast_state_online.py','runtime/continuous.py',
                                    'runtime/continuous_state.py','models/content_fast_state.py','evaluation/fast_state_online_demo.py'])}
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{ 'seed':r['seed'],'answers':r['answers'],'no_stale_answer':r['no_old_answer_after_correction']} for r in report['runs']]))


if __name__=='__main__':main()
