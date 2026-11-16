"""Shared event-stream routing reference: progress versus semantic staleness.

Entirely scripted entity/color backend and oracle entity dependencies. No neural
model/training/latency benchmark. A deliberately unsafe background-all control
shows why output progress cannot substitute for stale-assertion measurements.
"""
import asyncio
import json
from pathlib import Path

from runtime.continuous import ContinuousRunner
from runtime.dependency_router import DependencyRouter
from runtime.continuous_state import ExactReferenceExtractor
from evaluation.provenance import capture


class ScriptedAnswer:
    def __init__(self):self.parser=ExactReferenceExtractor()
    async def generate(self,request):
        color=0
        for _,text in request.inputs:
            result=self.parser(text)
            if result.get('accepted') and result['entity']==0:color=result['color']
        for index in range(60):
            for _ in range(3):await asyncio.sleep(0)
            yield json.dumps({'entity':0,'color':color,'step':index})


async def run_policy(policy):
    runner=ContinuousRunner(ScriptedAnswer(),history_limit=256)
    router=DependencyRouter(runner)
    await runner.observe('甲号设备的颜色是红。');router.set_dependencies({0})
    truth=0;checked=0;stale=0;during=0;trace=[]
    def score(during_stream):
        nonlocal checked,stale,during
        for committed in runner.committed[checked:]:
            value=json.loads(committed['unit']);wrong=value['color']!=truth
            stale+=int(wrong);during+=int(during_stream)
            trace.append({'generation_revision':committed['revision'],'color':value['color'],
                          'truth_at_commit_observed':truth,'stale':wrong,'during_inputs':during_stream})
        checked=len(runner.committed)
    try:
        for i in range(100):
            score(True)
            related=i==50
            text='把甲号设备的颜色更新为绿。' if related else '乙号设备的颜色是蓝。'
            if related:truth=2
            if policy=='hard_all':await runner.observe(text)
            elif policy=='dependency':
                await router.observe(text)
                if related:router.set_dependencies({0})
            else:await runner.observe_background(text,relevance_assertion='unsafe experimental negative control')
            # Fixed cooperative event-loop work schedule, not wall-clock benchmark.
            await asyncio.sleep(0)
        score(True)
        active_revision=runner._generation_revision
        await runner.wait_event(lambda e:e['kind']=='generation_finished' and e['generation_revision']==active_revision,timeout=5)
        score(False)
    finally:await runner.close()
    return {'policy':policy,'input_events':100,'related_corrections':1,
            'committed_during_input_stream':during,'total_committed':checked,
            'stale_assertions':stale,'assertion_trace':trace,
            'generation_starts':sum(e['kind']=='generation_started' for e in runner.journal),
            'cancellations':sum(e['kind']=='generation_cancelled' for e in runner.journal),
            'runtime_journal':runner.journal,'routing_journal':router.journal}


def main():
    target=Path('experiments/continuous_online/routing_comparison.json')
    if target.exists():raise FileExistsError(target)
    async def execute():return [await run_policy(p) for p in ('hard_all','dependency','background_all_unsafe')]
    runs=asyncio.run(execute())
    if runs[0]['stale_assertions'] or runs[1]['stale_assertions']:raise AssertionError('conservative policy stale output')
    if runs[1]['committed_during_input_stream']<=runs[0]['committed_during_input_stream']:raise AssertionError('no progress gain in controlled schedule')
    if not runs[2]['stale_assertions']:raise AssertionError('negative control failed to expose stale output')
    report={'scope':'scripted routing reference with oracle dependencies, NOT model semantic performance',
            'runs':runs,'provenance':capture(['runtime/continuous.py','runtime/dependency_router.py','evaluation/continuous_routing_comparison.py'])}
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('policy','committed_during_input_stream','total_committed','stale_assertions','generation_starts')} for r in runs]))


if __name__=='__main__':main()
