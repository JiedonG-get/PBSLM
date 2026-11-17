"""Recount span-guard paired predictions and threshold-shift coverage."""
import json
from pathlib import Path
import torch
from evaluation.text_key_transfer import counts


def main():
    source=Path('experiments/recurrent_state/text_align600/span_guard.json');out=source.with_name('span_guard_summary.json')
    if out.exists():raise FileExistsError(out)
    data=json.loads(source.read_text(encoding='utf-8'));pooled={};diagnostics=[]
    for run in data['runs']:
        recounted={}
        for row in run['traces']:
            target=torch.tensor(row['targets'])
            for arm,field in [('state','state_prediction'),('span_guard','guard_prediction')]:
                for group,idx in [('all',torch.arange(16)),('familiar',torch.arange(12)),('new_names',torch.arange(12,16))]:
                    key=f"{row['support']}/{arm}/{group}"
                    c=counts(torch.tensor(row[field])[idx],target[idx]);item=recounted.setdefault(key,{n:0 for n in c})
                    for n,v in c.items():item[n]+=v
            for novelty,reject in zip(row['novelty'],row['gate_reject']):
                if reject!=(novelty>run['threshold']):raise AssertionError('threshold decision mismatch')
        if recounted!=run['metrics']:raise AssertionError('prediction count mismatch')
        for key,c in recounted.items():
            item=pooled.setdefault(key,{n:0 for n in c})
            for n,v in c.items():item[n]+=v
        diagnostics.append({'seed':run['seed'],'threshold':run['threshold'],'calibration_known_keep':run['calibration_known_keep'],
              'support16_ranks':sorted({r['span_rank'] for r in run['traces'] if r['support']==16}),
              'support8_ranks':sorted({r['span_rank'] for r in run['traces'] if r['support']==8})})
    for c in pooled.values():
        c['known_accuracy']=c['known_correct']/c['known_n'] if c['known_n'] else None
        c['known_answer_coverage']=(c['known_n']-c['known_false_unknown'])/c['known_n'] if c['known_n'] else None
        c['unknown_false_accept_rate']=c['unknown_false_accept']/c['unknown_n'] if c['unknown_n'] else None
    out.write_text(json.dumps({'source':str(source),'diagnostics':diagnostics,'pooled':pooled},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(diagnostics,indent=2))
    for key,c in pooled.items():
        if key.endswith('/all'):print(key,c['known_accuracy'],c['known_answer_coverage'],c['unknown_false_accept_rate'])


if __name__=='__main__':main()
