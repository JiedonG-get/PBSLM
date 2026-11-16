"""Recount cumulative ridge traces and summarize against prior gradient reference."""
import json
from pathlib import Path
import statistics


def main():
    source=Path('experiments/consolidation/cumulative_ridge200.json')
    target=source.with_name('cumulative_ridge200_summary.json')
    if target.exists():raise FileExistsError(target)
    data=json.loads(source.read_text(encoding='utf-8'));arms={}
    for run in data['runs']:
        for arm in run['arms'].values():
            for stage in arm['stages']:
                for result in stage['evaluation'].values():
                    accuracy=sum(a==b for a,b in zip(result['predicted'],result['targets']))/len(result['targets'])
                    if abs(accuracy-result['accuracy'])>1e-8:raise AssertionError('trace mismatch')
    for name in ('trained_e','random_e','token_mean'):
        runs=[r['arms'][name] for r in data['runs']]
        arms[name]={}
        for split in ('train','held'):
            arms[name][f'final_{split}']=statistics.mean(r['stages'][-1]['evaluation'][f'group{g}/{split}']['accuracy'] for r in runs for g in range(3))
            arms[name][f'gradient_reference_final_{split}']=statistics.mean(r['gradient_reference_final'][f'group{g}/{split}']['accuracy'] for r in runs for g in range(3))
            arms[name][f'oldest_{split}_stage0']=statistics.mean(r['stages'][0]['evaluation'][f'group0/{split}']['accuracy'] for r in runs)
            arms[name][f'oldest_{split}_final']=statistics.mean(r['stages'][-1]['evaluation'][f'group0/{split}']['accuracy'] for r in runs)
        arms[name]['storage_final']=runs[0]['stages'][-1]['storage']
        arms[name]['max_reverse_weight_delta']=max(r['reverse_order_max_abs_weight_delta'] for r in runs)
    result={'source':str(source),'arms':arms,'note':'descriptive mean, different loss/state budget vs gradient'}
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
