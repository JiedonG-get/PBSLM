"""Paired original versus anchored/no-eviction write policy, frozen reader.

Random keys only, fixed.9 threshold. Clean initial priming followed by512 noisyhot0
writes. Adds one truly new ninth key after full capacity to expose lost acquisition.
No deferred-write queue, no history fallback. Targets always latest delivered write,
including deferred writes, so stale retained values are NOT scored as correct.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from models.dynamic_content_slots import DynamicContentSlots
from models.conservative_content_slots import ConservativeContentSlots
from runtime.fast_state_online import load_fast_model
from evaluation.continuous_key_control import state_hash
from evaluation.fast_state_geometry import geometry_keys,confusion
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(seed,noise):
    fast,artifact=load_fast_model(ROOT/f'experiments/recurrent_state/content_fast600/delta_seed{seed}.pt')
    before=state_hash(fast);models={'replace_evict':DynamicContentSlots(fast),'anchor_defer':ConservativeContentSlots(fast)}
    gen=torch.Generator().manual_seed(seed+241000);batch=16
    keys=geometry_keys(torch.randn(batch,9,16,generator=gen),'random')
    truth=torch.randint(8,(batch,8),generator=gen)
    # Consume original diagnostic's independent unknown draw to preserve hot stream.
    torch.randn(batch,16,generator=gen)
    states={name:model.initial(batch) for name,model in models.items()}
    for e in range(8):
        for name,model in models.items():states[name]=model.observe(states[name],keys[:,e],truth[:,e])
    counts={'accepted':0,'deferred':0};last_accepted=torch.zeros(batch,dtype=torch.bool)
    for _ in range(512):
        value=torch.randint(8,(batch,),generator=gen);eps=torch.randn(batch,16,generator=gen)
        noisy=F.normalize(keys[:,0]+noise*eps,dim=-1);truth[:,0]=value
        states['replace_evict']=models['replace_evict'].observe(states['replace_evict'],noisy,value)
        states['anchor_defer'],receipt=models['anchor_defer'].observe_receipt(states['anchor_defer'],noisy,value)
        counts['accepted']+=int(receipt['accepted'].sum());counts['deferred']+=int(receipt['deferred'].sum())
        last_accepted=receipt['accepted'].cpu()
    result={}
    for name,model in models.items():
        pred=torch.stack([model.query(states[name],keys[:,e]).argmax(-1) for e in range(8)],1)
        result[name]={'predictions':pred.tolist(),'targets':truth.tolist(),
                      'hot':confusion(pred[:,:1],truth[:,:1]),'cold':confusion(pred[:,1:],truth[:,1:])}
    ninth=F.normalize(torch.randn(batch,16,generator=gen),dim=-1);value=torch.randint(8,(batch,),generator=gen)
    new={}
    for name,model in models.items():
        if name=='anchor_defer':
            updated,receipt=model.observe_receipt(states[name],ninth,value);new_accepted=int(receipt['accepted'].sum())
        else:updated=model.observe(states[name],ninth,value);new_accepted=batch
        predicted=model.query(updated,ninth).argmax(-1)
        new[name]={'accepted':new_accepted,'n':batch,'correct':int((predicted==value).sum()),
                   'predicted':predicted.tolist(),'targets':value.tolist()}
    if before!=state_hash(fast):raise AssertionError('reader changed')
    return {'seed':seed,'write_noise':noise,'checkpoint':artifact,'hot_write_counts':counts,
           'last_hot_write_accepted':last_accepted.tolist(),'after_hot':result,'ninth_identity':new,'reader_unchanged':True}


def main():
    out=ROOT/'experiments/recurrent_state/content_fast600/write_admission512.json'
    if out.exists():raise FileExistsError(out)
    report={'scope':'rule-based anchored/no-eviction admission versus original slot policy, two factors changed',
      'provenance':capture(['models/conservative_content_slots.py','models/dynamic_content_slots.py','evaluation/write_admission_control.py']),
      'runs':[run(seed,noise) for seed in (0,1,2) for noise in (0.,.05,.15)]}
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for noise in (0.,.05,.15):
        rows=[r for r in report['runs'] if r['write_noise']==noise]
        print(noise,{name:{part:sum(r['after_hot'][name][part]['correct'] for r in rows)/sum(r['after_hot'][name][part]['n'] for r in rows)
                        for part in ('hot','cold')} for name in ('replace_evict','anchor_defer')},
               'accepted',sum(r['hot_write_counts']['accepted'] for r in rows)/24576)


if __name__=='__main__':main()
