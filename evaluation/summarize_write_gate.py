"""Recount write-gate probes and compare original soft control to prior skew run."""
import json
from pathlib import Path
import statistics


def main():
    source=Path('experiments/recurrent_state/write_gate8192.json');out=source.with_name('write_gate8192_summary.json')
    if out.exists():raise FileExistsError(out)
    data=json.loads(source.read_text(encoding='utf-8'))
    old=json.loads(Path('experiments/recurrent_state/skew8192.json').read_text(encoding='utf-8'))
    for run in data['runs']:
        prior=next(r for r in old['runs'] if r['seed']==run['seed'] and r['architecture']=='learned')
        for mode,rows in run['records'].items():
            for row in rows:
                predicted=row['predicted_ids'];target=row['target_ids']
                hot=sum(p[0]==t[0] for p,t in zip(predicted,target))
                cold=sum(a==b for p,t in zip(predicted,target) for a,b in zip(p[1:],t[1:]))
                if (hot,cold)!=(row['hot_correct'],row['cold_correct']):raise AssertionError('probe count mismatch')
            if rows[0]['predicted_ids']!=run['records']['soft'][0]['predicted_ids']:
                raise AssertionError('initial probes differ across gates')
        for a,b in zip(prior['records'],run['records']['soft']):
            if a['target_ids']!=b['target_ids'] or a['predicted_ids']!=b['predicted_ids']:
                raise AssertionError('original soft replay differs from recorded prior skew')
    result={'source':str(source),'original_soft_reproduces_prior_probes':True,'arms':{}}
    for mode in ('soft','top1_mass','top1_unit'):
        result['arms'][mode]=[{'step':row['step'],
          'cold_mean':statistics.mean(r['records'][mode][i]['cold_correct']/r['records'][mode][i]['cold_n'] for r in data['runs']),
          'hot_mean':statistics.mean(r['records'][mode][i]['hot_correct']/r['records'][mode][i]['hot_n'] for r in data['runs']),
          'cold_per_seed':[r['records'][mode][i]['cold_correct']/r['records'][mode][i]['cold_n'] for r in data['runs']]}
          for i,row in enumerate(data['runs'][0]['records'][mode])]
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({mode:rows[-1] for mode,rows in result['arms'].items()},indent=2))


if __name__=='__main__':main()
