"""Fixed recurrent-state cold-record retention under repeated hot-slot writes.

No training/history fallback. All8 IDs primed; only hotID0 updated thereafter.
Probe all entities at fixed checkpoints; truth used solely as external labels.
"""
import argparse
import json
from pathlib import Path
import torch
from evaluation.working_state_checkpoint import load_working_state
from evaluation.recurrent_state_control import weights_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINTS=(0,1,8,32,128,512,2048,8192)


@torch.no_grad()
def run(path,seed,kind,batch,device):
    model,metadata=load_working_state(path,device);before=weights_hash(model)
    if (metadata['seed'],metadata['architecture'])!=(seed,kind):raise ValueError('checkpoint filename/metadata mismatch')
    generator=torch.Generator().manual_seed(seed+121000)
    state=model.initial(batch);truth=torch.randint(8,(batch,8),generator=generator).to(device)
    # Same ascending priming order across models, no unseen queries.
    for entity in range(8):
        state=model.observe(state,torch.full((batch,),entity,device=device,dtype=torch.long),truth[:,entity])
    primed=state.clone();initial_truth=truth.clone();records=[]
    def probe(step):
        saved=state.clone();predictions=[]
        for entity in range(8):
            prediction=model.query(state,torch.full((batch,),entity,device=device,dtype=torch.long)).argmax(-1)
            predictions.append(prediction)
        predicted=torch.stack(predictions,1)
        if not torch.equal(saved,state):raise AssertionError('query changed working state')
        if not torch.isfinite(state).all():raise FloatingPointError('state nonfinite')
        diagnostic={'step':step,'hot_correct':int((predicted[:,0]==truth[:,0]).sum()),'hot_n':batch,
                    'cold_correct':int((predicted[:,1:]==truth[:,1:]).sum()),'cold_n':batch*7,
                    'predicted_ids':predicted.cpu().tolist(),'target_ids':truth.cpu().tolist(),
                    'state_change_l2_mean':float((state-primed).reshape(batch,-1).norm(dim=-1).mean())}
        if kind!='global':
            route=model.routing(torch.tensor([0],device=device))[0]
            selected=int(route.argmax());other=torch.arange(8,device=device)!=selected
            diagnostic['non_hot_top_slot_change_l2_mean']=float((state[:,other]-primed[:,other]).reshape(batch,-1).norm(dim=-1).mean())
            diagnostic['hot_top_slot']=selected
        records.append(diagnostic)
    probe(0)
    for step in range(1,max(CHECKPOINTS)+1):
        value=torch.randint(8,(batch,),generator=generator).to(device)
        truth[:,0]=value
        state=model.observe(state,torch.zeros(batch,device=device,dtype=torch.long),value)
        if step in CHECKPOINTS:probe(step)
    if not torch.equal(truth[:,1:],initial_truth[:,1:]):raise AssertionError('cold truth changed')
    if weights_hash(model)!=before:raise AssertionError('evaluation altered model parameters')
    reset_predictions=[]
    for entity in range(8):
        reset_predictions.append(model.query(torch.zeros_like(state),torch.full((batch,),entity,device=device,dtype=torch.long)).argmax(-1))
    reset=torch.stack(reset_predictions,1)
    return {'seed':seed,'architecture':kind,'checkpoint':metadata,'batch_streams':batch,
            'weights_before_sha256':before,'weights_after_sha256':weights_hash(model),
            'priming_order':list(range(8)),'hot_entity':0,'records':records,
            'reset_final':{'correct':int((reset==truth).sum()),'n':batch*8},
            'parameter_updates':0,'history_access':False,'note':'one fixed hot identity and priming order, not general skew robustness'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=32)
    p.add_argument('--output',default='experiments/recurrent_state/skew8192.json');args=p.parse_args()
    if args.batch<1:p.error('positive batch required')
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    report={'config':vars(args),'device':device,'checkpoints':CHECKPOINTS,'runs':[],
            'scope':'fixed-weight symbolic cold-record interference, no Online language capability',
            'provenance':capture(['evaluation/working_state_checkpoint.py','evaluation/working_state_skew.py',
                                  'models/recurrent_working_state.py','models/slotted_working_state.py'])}
    for seed in (0,1,2):
        for kind in ('global','learned','explicit'):
            result=run(ROOT/f'experiments/recurrent_state/slots600/{kind}_seed{seed}.pt',seed,kind,args.batch,device)
            report['runs'].append(result)
            print(json.dumps({'seed':seed,'architecture':kind,'cold_curve':[(r['step'],r['cold_correct']/r['cold_n']) for r in result['records']],
                              'final_hot':result['records'][-1]['hot_correct']/args.batch}),flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
