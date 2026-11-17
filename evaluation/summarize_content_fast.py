"""Aggregate paired associative-state results, checking counters and query pairing."""
import json
import statistics
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'experiments/recurrent_state/content_fast600/results.json'
    target=source.with_name('summary.json')
    if target.exists():raise FileExistsError(target)
    data=json.loads(source.read_text(encoding='utf-8'))
    if len(data['runs'])!=6:raise ValueError('all six runs required')
    original=json.loads((root/'experiments/recurrent_state/continuous_keys600/results.json').read_text(encoding='utf-8'))
    summary={}
    for run in data['runs']:
        reference=next(r for r in original['runs'] if r['seed']==run['seed'])
        for key,result in run['evaluation'].items():
            for group in ('known','unknown'):
                m=result['learned'][group]
                if abs(m['correct']/m['n']-m['accuracy'])>1e-12:raise AssertionError('count mismatch')
                if m['n']!=reference['evaluation'][key]['learned'][group]['n']:raise AssertionError('query groups not paired')
    for rule in ('delta','additive'):
        runs=[r for r in data['runs'] if r['rule']==rule]
        summary[rule]={'write_rates':[r['write_rate'] for r in runs],
            'parameters':[r['parameters'] for r in runs],
            'evaluation':{key:{group:{'mean':statistics.mean(r['evaluation'][key]['learned'][group]['accuracy'] for r in runs),
                           'per_seed':[r['evaluation'][key]['learned'][group]['accuracy'] for r in runs]}
                           for group in ('known','unknown')} for key in runs[0]['evaluation']}}
    report={'source':str(source.relative_to(root)),'arms':summary,'original_global_reference':{
        key:{group:statistics.mean(r['evaluation'][key]['learned'][group]['accuracy'] for r in original['runs'])
             for group in ('known','unknown')} for key in original['runs'][0]['evaluation']},
        'note':'macro three-seed descriptive means; no raw event-level prediction audit, source counters only'}
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for arm in summary:
        print(arm,json.dumps(summary[arm]['evaluation']['96/noise0.0/resetFalse']))


if __name__=='__main__':main()
