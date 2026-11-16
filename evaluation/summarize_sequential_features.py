"""Recount all sequential readout predictions and aggregate stage-wise retention."""
import json
from pathlib import Path
import statistics


def main():
    source=Path('experiments/consolidation/sequence_random_e200.json')
    target=source.with_name('sequence_random_e200_summary.json')
    if target.exists():raise FileExistsError(target)
    data=json.loads(source.read_text(encoding='utf-8'))
    for run in data['runs']:
        for arm in run['arms'].values():
            for ev in [arm['initial']]+[s['evaluation'] for s in arm['stages']]:
                for m in ev.values():
                    value=sum(a==b for a,b in zip(m['predicted'],m['targets']))/len(m['targets'])
                    if abs(value-m['accuracy'])>1e-6:raise AssertionError('trace mismatch')
    result={'source':str(source),'seeds':[r['seed'] for r in data['runs']],'arms':{}}
    for arm in ('trained_e','random_e','token_mean'):
        stages=[]
        for stage in range(3):
            metrics={key:statistics.mean(r['arms'][arm]['stages'][stage]['evaluation'][key]['accuracy'] for r in data['runs'])
                     for key in data['runs'][0]['arms'][arm]['stages'][stage]['evaluation']}
            stages.append({'stage':stage,'metrics':metrics})
        final=data['runs'][0]['arms'][arm]['stages'][-1]['evaluation']
        result['arms'][arm]={'stages':stages,
              'final_mean_train_all_groups':statistics.mean(r['arms'][arm]['stages'][-1]['evaluation'][f'group{g}/train']['accuracy'] for r in data['runs'] for g in range(3)),
              'final_mean_held_all_groups':statistics.mean(r['arms'][arm]['stages'][-1]['evaluation'][f'group{g}/held']['accuracy'] for r in data['runs'] for g in range(3)),
              'final_per_seed_held_all_groups':[statistics.mean(r['arms'][arm]['stages'][-1]['evaluation'][f'group{g}/held']['accuracy'] for g in range(3)) for r in data['runs']],
              'oldest_train_after_stage0':stages[0]['metrics']['group0/train'],
              'oldest_train_final':stages[-1]['metrics']['group0/train'],
              'oldest_held_after_stage0':stages[0]['metrics']['group0/held'],
              'oldest_held_final':stages[-1]['metrics']['group0/held']}
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
