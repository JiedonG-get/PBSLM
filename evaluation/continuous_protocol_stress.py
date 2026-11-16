"""Long deterministic protocol exercise; no neural quality/performance claims."""
import asyncio
import json
from pathlib import Path
import random

from runtime.continuous import ContinuousRunner
from evaluation.provenance import capture


class CooperativeBackend:
    def __init__(self):self.calls=0;self.live=0;self.peak=0
    async def generate(self,request):
        self.calls+=1;self.live+=1;self.peak=max(self.peak,self.live)
        try:
            for i in range(3):
                await asyncio.sleep(0)  # fair event-loop interleave, not measured model delay
                yield f'[{request.revision}:{i}]'
        finally:self.live-=1


async def exercise(seed,count):
    backend=CooperativeBackend();runner=ContinuousRunner(backend,history_limit=32,
                            max_committed_chars=1000000,max_generation_tasks=3)
    rng=random.Random(seed);peak_tasks=0;max_history=0
    try:
        for i in range(count):
            action=rng.randrange(10)
            if action==0 and runner.inputs:await runner.interrupt('stress_interrupt')
            elif action==1 and runner.inputs:await runner.resume()
            else:await runner.observe(f'event-{i}',should_respond=action>=4)
            peak_tasks=max(peak_tasks,sum(not task.done() for task in runner._tasks))
            max_history=max(max_history,len(runner.inputs))
            await asyncio.sleep(0)
        final=await runner.observe('final-latest-event',should_respond=True)
        await runner.wait_event(lambda e:e['kind']=='generation_finished' and e['generation_revision']==final,timeout=10)
    finally:await runner.close()
    stale=[e for e in runner.journal if e['kind']=='output_committed' and e['revision']!=e['generation_revision']]
    if stale or peak_tasks>3 or max_history>32 or backend.live or runner._tasks:
        raise AssertionError('continuous protocol bounds/lifecycle failed')
    if not any(x['revision']==final for x in runner.committed):raise AssertionError('latest intent never recovered')
    return {'seed':seed,'actions':count,'final_revision':final,'generation_calls':backend.calls,
            'peak_live_backend_generations':backend.peak,'peak_tracked_unfinished_tasks':peak_tasks,
            'max_recent_input_history':max_history,'committed_units':len(runner.committed),
            'stale_commits':len(stale),'tasks_after_close':len(runner._tasks),
            'final_latest_request_completed':True,'pending_after_close':runner._pending_revision,
            'event_counts':{kind:sum(e['kind']==kind for e in runner.journal)
                             for kind in sorted({e['kind'] for e in runner.journal})}}


def main():
    target=Path('experiments/continuous_online/protocol_stress1000.json')
    if target.exists():raise FileExistsError(target)
    async def run():return [await exercise(seed,1000) for seed in (0,1,2)]
    report={'scope':'cooperative scripted protocol stress, NOT neural semantic/throughput benchmark',
            'runs':asyncio.run(run()),'no_training':True,
            'provenance':capture(['runtime/continuous.py','evaluation/continuous_protocol_stress.py'])}
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report['runs'],indent=2))


if __name__=='__main__':main()
