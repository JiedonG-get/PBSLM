"""Recount aligned projection probes and report familiar/new identity groups."""
import json
from pathlib import Path
import torch
from evaluation.text_key_transfer import counts


def main():
    source=Path('experiments/recurrent_state/text_align600/results.json');out=source.with_name('summary.json')
    if out.exists():raise FileExistsError(out)
    data=json.loads(source.read_text(encoding='utf-8'));pooled={}
    for run in data['runs']:
        for arm,ev in run['evaluations'].items():
            recounted={}
            for row in ev['traces']:
                for method,predictions in row['predictions'].items():
                    for group,indices in [('familiar',list(range(12))),('new_names',list(range(12,16))),('all',list(range(16)))]:
                        key=f"{row['support']}/{method}/{group}"
                        c=counts(torch.tensor(predictions)[indices],torch.tensor(row['targets'])[indices])
                        item=recounted.setdefault(key,{n:0 for n in c})
                        for n,v in c.items():item[n]+=v
            if recounted!=ev['metrics']:raise AssertionError('prediction/metric mismatch')
            for key,c in recounted.items():
                item=pooled.setdefault(arm+'/'+key,{n:0 for n in c})
                for n,v in c.items():item[n]+=v
        random=run['evaluations']['random']['traces'];aligned=run['evaluations']['aligned']['traces']
        for a,b in zip(random,aligned):
            for name in ('records','support_templates','query_templates','written_values','targets'):
                if a[name]!=b[name]:raise AssertionError('unpaired evaluation events')
    for item in pooled.values():
        item['known_accuracy']=item['known_correct']/item['known_n'] if item['known_n'] else None
        item['unknown_false_accept_rate']=item['unknown_false_accept']/item['unknown_n'] if item['unknown_n'] else None
    out.write_text(json.dumps({'source':str(source),'pooled':pooled},indent=2)+'\n',encoding='utf-8')
    for key,item in pooled.items():
        if '/8/' in key and ('/fast/' in key or '/nearest_always/' in key):
            print(key,item['known_accuracy'],item['unknown_false_accept_rate'])


if __name__=='__main__':main()
