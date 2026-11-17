"""Recount frozen-state probe traces and summarize skew-retention curves."""
import json
from pathlib import Path
import statistics


def main():
    source=Path('experiments/recurrent_state/skew8192.json');out=source.with_name('skew8192_summary.json')
    if out.exists():raise FileExistsError(out)
    data=json.loads(source.read_text(encoding='utf-8'))
    for run in data['runs']:
        for probe in run['records']:
            hot=sum(p[0]==t[0] for p,t in zip(probe['predicted_ids'],probe['target_ids']))
            cold=sum(a==b for p,t in zip(probe['predicted_ids'],probe['target_ids']) for a,b in zip(p[1:],t[1:]))
            if (hot,cold)!=(probe['hot_correct'],probe['cold_correct']):raise AssertionError('probe count mismatch')
        if run['weights_before_sha256']!=run['weights_after_sha256']:raise AssertionError('weight hash changed')
    result={'source':str(source),'arms':{},'note':'same batch size per seed; seed mean descriptive, checkpoints highly correlated'}
    for kind in ('global','learned','explicit'):
        runs=[r for r in data['runs'] if r['architecture']==kind]
        result['arms'][kind]=[{'step':step,
            'cold_accuracy_mean':statistics.mean(r['records'][i]['cold_correct']/r['records'][i]['cold_n'] for r in runs),
            'cold_per_seed':[r['records'][i]['cold_correct']/r['records'][i]['cold_n'] for r in runs],
            'hot_accuracy_mean':statistics.mean(r['records'][i]['hot_correct']/r['records'][i]['hot_n'] for r in runs)}
            for i,step in enumerate(data['checkpoints'])]
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
