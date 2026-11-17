"""Frozen dynamic-slot/fast-state stress with noisy WRITES and clean queries.

Semantic identity in evaluator only; slots get continuous noisy key/value, not ID.
Slot ownership diagnostics are counterfactual provenance tags, never model inputs.
Fixed512hot0 writes, capacity8; not long-run universal or language robustness.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from models.dynamic_content_slots import DynamicContentSlots
from runtime.fast_state_online import load_fast_model
from evaluation.fast_state_geometry import geometry_keys,confusion
from evaluation.continuous_key_control import state_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(seed,geometry,noise):
    fast,artifact=load_fast_model(ROOT/f'experiments/recurrent_state/content_fast600/delta_seed{seed}.pt')
    before=state_hash(fast);slots=DynamicContentSlots(fast,capacity=8,threshold=.9)
    gen=torch.Generator().manual_seed(seed+241000);batch=16
    keys=geometry_keys(torch.randn(batch,9,16,generator=gen),geometry)
    truth=torch.randint(8,(batch,8),generator=gen)
    unknown=F.normalize(torch.randn(batch,16,generator=gen),dim=-1)
    matrix=fast.initial(batch);state=slots.initial(batch)
    owners=torch.full((batch,8),-1,dtype=torch.long)
    counters={phase:{'writes':0,'wrong_merge':0,'duplicate_allocation':0,'eviction':0,'cold_eviction':0}
              for phase in ('priming','hot')}
    def write(key,value,entity,phase):
        nonlocal state,matrix,owners
        # Actual allocation does not receive entity or owners. Diagnostics below
        # compare its decisions to semantic writer labels held by evaluator.
        _,score,index=slots._match(state,key)
        occupied=state[:,:,24]>.5
        diagnostics=counters[phase]
        for b in range(batch):
            diagnostics['writes']+=1
            if score[b]>=slots.threshold:
                diagnostics['wrong_merge']+=int(owners[b,index[b]]!=entity)
            else:
                diagnostics['duplicate_allocation']+=int((owners[b]==entity).any())
                if occupied[b].all():
                    diagnostics['eviction']+=1
                    old=int(state[b,:,25].argmin())
                    diagnostics['cold_eviction']+=int(owners[b,old]!=entity)
        updated=slots.observe(state,key,value)
        # Newest rank is unique among occupied slots under normal state invariants.
        selected=updated[:,:,25].masked_fill(updated[:,:,24]<.5,-1).argmax(-1)
        owners=owners.clone();owners[torch.arange(batch),selected]=entity
        state=updated;matrix=fast.observe(matrix,key,value)
    for entity in range(8):write(keys[:,entity],truth[:,entity],entity,'priming')
    records=[]
    def probe(step):
        matrix_before=matrix.clone();slot_before=state.clone()
        predictions={method:torch.stack([model.query(h,keys[:,e]).argmax(-1) for e in range(8)],1)
                     for method,model,h in [('fast',fast,matrix),('slots',slots,state)]}
        rows={}
        for method,model,h in [('fast',fast,matrix),('slots',slots,state)]:
            absent=model.query(h,unknown).argmax(-1);p=predictions[method]
            rows[method]={'hot':confusion(p[:,:1],truth[:,:1]),'cold':confusion(p[:,1:],truth[:,1:]),
                  'predicted':p.tolist(),'unknown_predictions':absent.tolist(),
                  'unknown_false_accept':int((absent!=8).sum()),'unknown_n':batch}
        if not torch.equal(matrix,matrix_before) or not torch.equal(state,slot_before):raise AssertionError('query changed state')
        records.append({'step':step,'targets':truth.tolist(),'methods':rows,
                        'occupied_slots_per_stream':(state[:,:,24]>.5).sum(-1).tolist(),
                        'distinct_current_owners':[len(set(row.tolist())-{-1}) for row in owners]})
    probe(0)
    for step in range(1,513):
        values=torch.randint(8,(batch,),generator=gen)
        eps=torch.randn(batch,16,generator=gen)
        key=F.normalize(keys[:,0]+noise*eps,dim=-1);truth[:,0]=values
        write(key,values,0,'hot')
        if step in (1,32,128,512):probe(step)
    if before!=state_hash(fast):raise AssertionError('parameters changed')
    return {'seed':seed,'geometry':geometry,'write_noise':noise,'checkpoint':artifact,'records':records,
             'slot_admission_diagnostics':counters,'weights_unchanged':True,
             'semantics':'owner labels for diagnostics only, all queries clean exact original identities'}


def main():
    out=ROOT/'experiments/recurrent_state/content_fast600/write_noise512.json'
    if out.exists():raise FileExistsError(out)
    report={'scope':'fixed model/state algorithms, write-noise identity ambiguity, not a language test',
            'provenance':capture(['evaluation/write_noise_control.py','models/dynamic_content_slots.py',
                                 'models/content_fast_state.py','evaluation/fast_state_geometry.py']), 'runs':[]}
    for seed in (0,1,2):
        for geometry in ('random','correlated'):
            for noise in (0.,.05,.15):
                r=run(seed,geometry,noise);report['runs'].append(r)
                print(json.dumps({'seed':seed,'geometry':geometry,'noise':noise,'final':{
                 m:{'hot':v['hot']['correct']/v['hot']['n'],'cold':v['cold']['correct']/v['cold']['n']}
                   for m,v in r['records'][-1]['methods'].items()},'admission':r['slot_admission_diagnostics']['hot']}),flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
