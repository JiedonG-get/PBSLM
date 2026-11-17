"""Recount query-boundary saved predictions; no model calls."""
import json
from pathlib import Path
from evaluation.fast_query_boundary import confusion
import torch


def main():
    source=Path('experiments/recurrent_state/content_fast600/query_boundary.json')
    out=source.with_name('query_boundary_summary.json')
    if out.exists():raise FileExistsError(out)
    data=json.loads(source.read_text(encoding='utf-8'));summary={}
    for run in data['runs']:
        for row in run['conditions']:
            for method,result in row['methods'].items():
                counts=confusion(torch.tensor(result['predicted']),torch.tensor(row['targets']))
                if counts!=result['counts']:raise AssertionError('prediction mismatch')
                key=f"{run['geometry']}/{row['condition']}/{method}"
                pooled=summary.setdefault(key,{name:0 for name in counts})
                for name,value in counts.items():pooled[name]+=value
    for result in summary.values():
        result['known_accuracy']=result['known_correct']/result['known_n'] if result['known_n'] else None
        result['unknown_false_accept_rate']=result['unknown_false_accept']/result['unknown_n'] if result['unknown_n'] else None
    out.write_text(json.dumps({'source':str(source),'pooled':summary},indent=2)+'\n',encoding='utf-8')
    for geometry in ('random','correlated'):
        for condition in ('known_noise0','known_noise.05','known_noise.15','unknown_random','unknown_near.95','unknown_near.99'):
            print(geometry,condition,{m:(summary[f'{geometry}/{condition}/{m}']['known_accuracy'] if condition.startswith('known') else summary[f'{geometry}/{condition}/{m}']['unknown_false_accept_rate']) for m in ('delta','dual','nearest_cos.9')})


if __name__=='__main__':main()
