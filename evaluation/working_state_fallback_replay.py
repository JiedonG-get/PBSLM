"""Offline counterfactual fallback replay using saved neural predictions.

Reconstructs OBSERVED writes from original dedicated RNG, not target arrays. State
predictions frozen from earlier run, fallback never changes those states. Latest
value dictionary receives all writes, uses caller-provided entity IDs, not language
retrieval. Costs explicitly count updates/queries; no neural speed-up claim.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import torch


def observed_final(seed,hot,capacity,steps=2048,batch=16):
    gen=torch.Generator().manual_seed(seed+131000+hot*100)
    initial=torch.randint(8,(batch,8),generator=gen)
    ledgers=[OrderedDict() for _ in range(batch)]
    def write(b,key,value):
        ledger=ledgers[b];ledger.pop(key,None);ledger[key]=value
        if len(ledger)>capacity:ledger.popitem(last=False)
    for entity in range(8):
        for b in range(batch):write(b,entity,int(initial[b,entity]))
    last=initial.clone()
    for _ in range(steps):
        values=torch.randint(8,(batch,),generator=gen);last[:,hot]=values
        for b in range(batch):write(b,hot,int(values[b]))
    return ledgers,last


def apply_fallback(predictions,ledgers,top_slots):
    if len(predictions)!=len(ledgers) or len(top_slots)!=8:
        raise ValueError('matching prediction/ledger streams and eight routing entries required')
    if any(len(row)!=8 for row in predictions):raise ValueError('eight predictions per stream required')
    flagged={e for e in range(8) if top_slots.count(top_slots[e])>1}
    answers=[];queries=hits=0
    for predicted,ledger in zip(predictions,ledgers):
        row=[]
        for entity,answer in enumerate(predicted):
            if entity in flagged:
                queries+=1;answer=ledger.get(entity)
                hits+=int(answer is not None)
            row.append(answer)
        answers.append(row)
    return answers,{'queries':queries,'hits':hits,'misses':queries-hits}


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'experiments/recurrent_state/sparse_train600/results.json'
    out=root/'experiments/recurrent_state/fallback_replay.json'
    if out.exists():raise FileExistsError(out)
    data=json.loads(source.read_text(encoding='utf-8'));results=[]
    for run in data['runs']:
        for capacity in (4,8):
            total=correct=answered=queries=hits=0;cases=[]
            for case in run['all_hot']:
                ledgers,reconstructed=observed_final(run['seed'],case['hot'],capacity)
                # Verification only: neither saved target nor reconstructed truth enters
                # selection/lookup. Ledger built solely by observed_final write events.
                if reconstructed.tolist()!=case['final']['targets']:raise AssertionError('observed stream replay differs')
                answers,cost=apply_fallback(case['final']['predicted'],ledgers,run['top_slots'])
                for row,target in zip(answers,case['final']['targets']):
                    for answer,truth in zip(row,target):
                        total+=1;answered+=int(answer is not None);correct+=int(answer==truth)
                queries+=cost['queries'];hits+=cost['hits']
                cases.append({'hot':case['hot'],'answers':answers,'fallback_cost':cost,
                              'retained_ids_per_stream':[list(x) for x in ledgers]})
            results.append({'seed':run['seed'],'arm':run['arm'],'capacity':capacity,'n':total,
                            'correct':correct,'answered':answered,'accuracy_all':correct/total,
                            'coverage':answered/total,'fallback_queries':queries,'fallback_hits':hits,
                            'fallback_misses':queries-hits,'observed_write_updates':8*16*(8+2048),
                            'retained_pairs_per_stream':capacity,
                            'numeric_payload_lower_bound_bytes_per_stream':capacity*2*8,
                            'cases':cases})
    summary={}
    for arm in ('soft','sparse','sparse_collision'):
        for capacity in (4,8):
            rows=[r for r in results if r['arm']==arm and r['capacity']==capacity]
            n=sum(r['n'] for r in rows)
            summary[f'{arm}/{capacity}']={k:sum(r[k] for r in rows)/n for k in ('correct','answered','fallback_queries','fallback_misses')}
    report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'runs':results,'pooled_ratios':summary,
            'scope':'counterfactual saved-state prediction plus actual observed-write dictionary replay, not live integrated fallback',
            'limits':['fallback does not repair model state','all writes stored/updated even when never queried',
                      'explicit ID and exact observed labels, no semantic extraction','capacity4 miss returns abstention',
                      'payload lower bound excludes Python objects/indexes/metadata','no elapsed time or compute-saving evidence']}
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
