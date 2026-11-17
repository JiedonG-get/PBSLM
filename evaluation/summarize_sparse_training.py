"""Recount sparse training all-hot predictions; macro means and collision strata."""
import json
from pathlib import Path
import statistics


def main():
    source=Path('experiments/recurrent_state/sparse_train600/results.json');out=source.with_name('summary.json')
    if out.exists():raise FileExistsError(out)
    data=json.loads(source.read_text(encoding='utf-8'))
    if len(data['runs'])!=9:raise ValueError('all9 runs required')
    for run in data['runs']:
        for case in run['all_hot']:
            for when in ('initial','final'):
                probe=case[when]
                counts=[sum(p[e]==t[e] for p,t in zip(probe['predicted'],probe['targets'])) for e in range(8)]
                if counts!=probe['correct_per_entity']:raise AssertionError('prediction count mismatch')
    summary={}
    for arm in ('soft','sparse','sparse_collision'):
        runs=[r for r in data['runs'] if r['arm']==arm]
        perseed=[]
        for run in runs:
            hot=[];cold=[];colliding=noncolliding=coll_n=non_n=0
            for case in run['all_hot']:
                h=case['hot'];probe=case['final'];n=probe['n_per_entity'];counts=probe['correct_per_entity']
                hot.append(counts[h]/n);cold.append(sum(counts[e] for e in range(8) if e!=h)/(7*n))
                for e in range(8):
                    if e==h:continue
                    if run['top_slots'][e]==run['top_slots'][h]:colliding+=counts[e];coll_n+=n
                    else:noncolliding+=counts[e];non_n+=n
            perseed.append({'seed':run['seed'],'uniform384':run['uniform']['384']['accuracy'],
                            'all_hot':statistics.mean(hot),'all_cold':statistics.mean(cold),
                            'distinct_top_slots':run['distinct_top_slots'],
                            'colliding':{'correct':colliding,'n':coll_n},'noncolliding':{'correct':noncolliding,'n':non_n}})
        summary[arm]={'per_seed':perseed,'means':{key:statistics.mean(r[key] for r in perseed) for key in ('uniform384','all_hot','all_cold')},
                      'prior_final':[r['curve'][-1]['collision'] for r in runs],
                      'note':'top-slot collisions do not prove shared full representation, read remains soft'}
    out.write_text(json.dumps({'source':str(source),'arms':summary},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
