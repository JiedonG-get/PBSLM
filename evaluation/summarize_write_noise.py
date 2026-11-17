"""Recount noisy-write probes and pooled admission diagnostics, no model calls."""
import json
from pathlib import Path
from evaluation.fast_state_geometry import confusion
import torch


def main():
    source=Path('experiments/recurrent_state/content_fast600/write_noise512.json')
    out=source.with_name('write_noise512_summary.json')
    if out.exists():raise FileExistsError(out)
    data=json.loads(source.read_text(encoding='utf-8'));summary={}
    for run in data['runs']:
        for probe in run['records']:
            targets=torch.tensor(probe['targets'])
            for m,result in probe['methods'].items():
                prediction=torch.tensor(result['predicted'])
                for part,indices in [('hot',[0]),('cold',list(range(1,8)))]:
                    if confusion(prediction[:,indices],targets[:,indices])!=result[part]:raise AssertionError('prediction count mismatch')
                if sum(x!=8 for x in result['unknown_predictions'])!=result['unknown_false_accept']:
                    raise AssertionError('unknown count mismatch')
        key=f"{run['geometry']}/noise{run['write_noise']}"
        pooled=summary.setdefault(key,{'methods':{},'slot_diagnostics':{}})
        for phase,counts in run['slot_admission_diagnostics'].items():
            group=pooled['slot_diagnostics'].setdefault(phase,{name:0 for name in counts})
            for name,count in counts.items():group[name]+=count
        for m,result in run['records'][-1]['methods'].items():
            group=pooled['methods'].setdefault(m,{'hot_n':0,'hot_correct':0,'cold_n':0,'cold_correct':0,'unknown_n':0,'unknown_false_accept':0})
            for part in ('hot','cold'):
                group[part+'_n']+=result[part]['n'];group[part+'_correct']+=result[part]['correct']
            group['unknown_n']+=result['unknown_n'];group['unknown_false_accept']+=result['unknown_false_accept']
    for pooled in summary.values():
        for c in pooled['methods'].values():
            c['hot_accuracy']=c['hot_correct']/c['hot_n'];c['cold_accuracy']=c['cold_correct']/c['cold_n']
            c['unknown_false_accept_rate']=c['unknown_false_accept']/c['unknown_n']
    out.write_text(json.dumps({'source':str(source),'pooled':summary},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
