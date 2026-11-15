"""Recompute paired text recall descriptive means from raw immutable traces."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',default='experiments/text_facts/paired800')
    args=p.parse_args(); directory=Path(args.directory)
    summary={}
    for arm in ('baseline','key_aligned'):
        data=json.loads((directory/f'{arm}.json').read_text(encoding='utf-8'))
        for run in data['runs']:
            computed={}
            for trace in run['evaluation']['traces']:
                name=f"{trace['split']}/{trace['query']}/{trace['mode']}"
                acc=computed.setdefault(name,[0,0,0])
                acc[0]+=sum(a==b for a,b in zip(trace['a_ids'],trace['target_ids']))
                acc[1]+=len(trace['target_ids']); acc[2]+=len(set(trace['a_ids']))
            for name,values in computed.items():
                metric=run['evaluation']['metrics'][name]
                if metric['correct']!=values[0] or metric['n']!=values[1]:
                    raise AssertionError(f'trace/metric mismatch {arm} seed{run["seed"]} {name}')
        summary[arm]={'seeds':[r['seed'] for r in data['runs']],
                      'seconds':sum(r['total_seconds'] for r in data['runs']),
                      'metrics':{name:{'mean_accuracy':statistics.mean(r['evaluation']['metrics'][name]['accuracy'] for r in data['runs']),
                                       'per_seed':[r['evaluation']['metrics'][name]['accuracy'] for r in data['runs']],
                                       'mean_unique_predictions':statistics.mean(r['evaluation']['metrics'][name]['mean_unique_predictions_per_episode'] for r in data['runs'])}
                                 for name in data['runs'][0]['evaluation']['metrics']}}
    out=directory/'summary.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    for arm,values in summary.items():
        print(arm,'seconds',values['seconds'])
        for split in ('familiar','novel_entities'):
            for query in ('same_surface','seen_paraphrase','held_paraphrase'):
                print(split,query,{mode:values['metrics'][f'{split}/{query}/{mode}']['mean_accuracy'] for mode in ('normal','zero','wrong')})


if __name__=='__main__': main()
