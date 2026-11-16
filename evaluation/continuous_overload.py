"""Deterministic overload contract: bounded admission, no silent input loss.

A gated exact parser replaces model compute. Not a neural throughput benchmark.
No sleeps used to decide when ingestion is blocked; producer intentionally bursts.
"""
import asyncio
import json
from pathlib import Path
import threading

from runtime.continuous import ContinuousRunner
from runtime.continuous_state import ContinuousStateBackend,ExactReferenceExtractor
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


async def execute():
    entered=threading.Event();release=threading.Event();exact=ExactReferenceExtractor()
    def gated(text):
        entered.set()
        if not release.wait(10):raise TimeoutError('controlled ingest gate timed out')
        return exact(text)
    backend=ContinuousStateBackend(gated,queue_limit=4,snapshot_limit=8,unit_delay=0)
    runner=ContinuousRunner(backend)
    accepted=[];rejected=[];retried=[]
    try:
        first=await runner.observe('甲号设备的颜色是红。',should_respond=False)
        accepted.append({'revision':first,'text':'甲号设备的颜色是红。'})
        if not await asyncio.to_thread(entered.wait,2):raise AssertionError('worker not started')
        texts=['把甲号设备的颜色更新为蓝。','乙号设备的颜色是绿。',
               '把甲号设备的颜色更新为黄。','丙号设备的颜色是黑。',
               '把乙号设备的颜色更新为白。','把甲号设备的颜色更新为紫。']
        for text in texts:
            before=(runner.revision,tuple(runner.inputs),len(backend.futures))
            try:
                rev=await runner.observe(text,should_respond=False)
                accepted.append({'revision':rev,'text':text})
            except asyncio.QueueFull:
                if before!=(runner.revision,tuple(runner.inputs),len(backend.futures)):
                    raise AssertionError('rejected input mutated admission state')
                rejected.append({'text':text,'reason':'queue_full','runner_revision_unchanged':True})
        queued_at_gate=backend.queue.qsize();pending_at_gate=len(backend.futures)
        if queued_at_gate!=4 or len(rejected)!=2:raise AssertionError('queue capacity not respected')
        await runner.interrupt('producer-overload')
        release.set()
        await asyncio.wait_for(backend.queue.join(),5)
        # Producer owns rejected items: retry explicitly, preserving original order.
        for item in rejected:
            rev=await runner.observe(item['text'],should_respond=False)
            state=await backend.processed(rev)
            retried.append({'revision':rev,'text':item['text']})
        final=await backend.processed(retried[-1]['revision'])
        finished=[x['revision'] for x in backend.journal if x['kind']=='understanding_finished']
        expected=[x['revision'] for x in accepted+retried]
        if finished!=expected:raise AssertionError('accepted inputs not processed in admission order')
        if final[0]['color']!=6 or final[1]['color']!=5:raise AssertionError('latest retried update lost')
        if runner.output:raise AssertionError('silent overload emitted output')
    finally:
        release.set()
        try:await runner.close()
        finally:await backend.aclose()
    return {'scope':'thread-gated admission correctness, not model latency or semantic quality',
            'queue_limit':4,'queued_while_worker_blocked':queued_at_gate,
            'pending_including_active_worker':pending_at_gate,'accepted_before_release':accepted,
            'explicit_rejections':rejected,'producer_retries':retried,'processed_revisions':finished,
            'final_state':{str(k):v for k,v in final.items()},'no_silent_loss_of_accepted_inputs':True,
            'rejected_inputs_require_producer_retry':True,'no_output_from_passive_inputs':True,
            'runtime_journal':runner.journal,'ingest_journal':backend.journal}


def main():
    out=ROOT/'experiments/continuous_online/overload_admission.json'
    if out.exists():raise FileExistsError(out)
    report=asyncio.run(execute())
    report['provenance']=capture(['runtime/continuous.py','runtime/continuous_state.py','evaluation/continuous_overload.py'])
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('queue_limit','queued_while_worker_blocked',
                     'pending_including_active_worker','no_silent_loss_of_accepted_inputs')}))


if __name__=='__main__':main()
