"""Recount dual-key observations; no models or training."""
import json
from pathlib import Path


def main():
    source=Path('experiments/recurrent_state/content_fast600/dual_geometry512.json')
    out=source.with_name('dual_geometry512_summary.json')
    if out.exists():raise FileExistsError(out)
    data=json.loads(source.read_text(encoding='utf-8'))
    for run in data['runs']:
        for row in run['records']:
            for arm,key in [('delta','delta_predictions'),('dual','dual_predictions')]:
                for part,indices in [('hot',[0]),('cold',range(1,8))]:
                    pairs=[(p[i],t[i]) for p,t in zip(row[key],row['targets']) for i in indices]
                    c={'n':len(pairs),'correct':sum(a==b for a,b in pairs),'false_unknown':sum(a==8 for a,b in pairs),
                       'wrong_value':sum(a!=b and a!=8 for a,b in pairs)}
                    if c!=row[f'{arm}_{part}']:raise AssertionError('prediction count mismatch')
    summary={}
    for geometry in ('orthogonal','random','correlated'):
        for schedule in ('independent','initially_same'):
            runs=[r for r in data['runs'] if r['geometry']==geometry and r['value_schedule']==schedule]
            result={}
            for stage,index in [('initial',0),('final',-1)]:
                for arm in ('delta','dual'):
                    for part in ('hot','cold'):
                        rows=[r['records'][index][f'{arm}_{part}'] for r in runs]
                        result[f'{stage}/{arm}/{part}']=sum(x['correct'] for x in rows)/sum(x['n'] for x in rows)
            result['maximum_dual_read_error']=max(row['dual_read_max_error'] for r in runs for row in r['records'])
            result['reference_payload_bytes']=sum(runs[0]['reference_storage_per_stream'].values())
            summary[f'{geometry}/{schedule}']=result
    out.write_text(json.dumps({'source':str(source),'pooled':summary},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
