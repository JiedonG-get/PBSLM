"""Out-of-order source sequence reference evidence, no model/learning."""
import json
from pathlib import Path
import random
from runtime.source_order import SourceOrderedState


def main():
    out=Path('experiments/continuous_online/source_order.json')
    if out.exists():raise FileExistsError(out)
    runs=[]
    for seed in (0,1,2):
        events=[{'id':f'e{i}','sequence':i,'value':('red','green','blue')[i%3]} for i in range(30)]
        random.Random(seed).shuffle(events)
        state=SourceOrderedState();maximum=-1;trace=[]
        for event in events:
            receipt=state.apply(event['id'],source='simulated-stream',slot='lamp',
                                sequence=event['sequence'],value=event['value'])
            maximum=max(maximum,event['sequence']);view=state.view('lamp')
            if view['sources']['simulated-stream']['sequence']!=maximum:
                raise AssertionError('source watermark reverted')
            trace.append({'arrival':event,'receipt':receipt,'state':view})
        runs.append({'seed':seed,'trace':trace,'final_sequence':maximum,
                     'late_events_ignored':sum(x['receipt']['reason']=='late_ignored' for x in trace)})
    out.write_text(json.dumps({'scope':'isolated source-order reference, not integrated sensor/runtime timestamp policy',
                               'model_executed':False,'runs':runs},indent=2)+'\n',encoding='utf-8')
    print([(x['seed'],x['final_sequence'],x['late_events_ignored']) for x in runs])


if __name__=='__main__':main()
