"""Recount frozen geometry stress probes without model execution."""
import json
from pathlib import Path


def main():
    source=Path('experiments/recurrent_state/content_fast600/geometry_hot2048.json')
    target=source.with_name('geometry_hot2048_summary.json')
    if target.exists():raise FileExistsError(target)
    data=json.loads(source.read_text(encoding='utf-8'));summary={}
    for run in data['runs']:
        for row in run['records']:
            for part,indices in [('hot',[0]),('cold',range(1,8))]:
                pairs=[(p[i],t[i]) for p,t in zip(row['predicted'],row['targets']) for i in indices]
                counts={'n':len(pairs),'correct':sum(a==b for a,b in pairs),
                        'false_unknown':sum(a==8 for a,b in pairs),
                        'wrong_value':sum(a!=b and a!=8 for a,b in pairs)}
                if counts!=row[part]:raise AssertionError('confusion mismatch')
            if sum(p!=8 for p in row['absent_predictions'])!=row['unknown']['false_accept']:
                raise AssertionError('absent count mismatch')
    for rule in ('delta','additive'):
        for geometry in ('orthogonal','random','correlated'):
            runs=[r for r in data['runs'] if r['rule']==rule and r['geometry']==geometry]
            stages=[]
            for index in range(len(runs[0]['records'])):
                item={'step':runs[0]['records'][index]['step']}
                for part in ('hot','cold','unknown'):
                    counts={key:sum(r['records'][index][part][key] for r in runs) for key in runs[0]['records'][index][part]}
                    item[part]={**counts,'accuracy':counts.get('correct',counts.get('correct_reject'))/counts['n']}
                stages.append(item)
            summary[f'{rule}/{geometry}']=stages
    target.write_text(json.dumps({'source':str(source),'pooled_curves':summary},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{'initial_cold':v[0]['cold']['accuracy'],'final':v[-1]} for k,v in summary.items()},indent=2))


if __name__=='__main__':main()
