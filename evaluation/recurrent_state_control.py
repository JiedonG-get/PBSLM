"""Learn fixed-state updates on symbolic streams, no inference history access.

Evaluator maintains truth labels, never passed to recurrent query. Observations are
explicit entity/value writes. Training BPTT stores activations, evaluation only h.
No language, adaptive fallback, PBSLM integration or holographic claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from models.recurrent_working_state import RecurrentWorkingState
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


def weights_hash(model):
    h=hashlib.sha256()
    for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def stream(model,generator,batch,steps,device,train=False,reset_query=False):
    state=model.initial(batch);truth=torch.full((batch,8),-1,dtype=torch.long,device=device)
    losses=[];correct=0;total=0;by_age={};last=torch.full_like(truth,-1)
    for step in range(steps):
        entity=torch.randint(8,(batch,),generator=generator).to(device)
        value=torch.randint(8,(batch,),generator=generator).to(device)
        state=model.observe(state,entity,value)
        truth[torch.arange(batch,device=device),entity]=value
        last[torch.arange(batch,device=device),entity]=step
        # Query a previously seen entity, sampled independently of current write.
        scores=torch.rand(batch,8,generator=generator).to(device)
        scores[truth<0]=-1;query=scores.argmax(-1)
        target=truth[torch.arange(batch,device=device),query]
        output=model.query(torch.zeros_like(state) if reset_query else state,query)
        losses.append(F.cross_entropy(output,target))
        prediction=output.argmax(-1);hits=prediction==target
        correct+=int(hits.sum());total+=batch
        if not train:
            ages=step-last[torch.arange(batch,device=device),query]
            for age,hit in zip(ages.tolist(),hits.tolist()):
                bucket='0' if age==0 else '1-3' if age<=3 else '4-15' if age<=15 else '16+'
                item=by_age.setdefault(bucket,{'correct':0,'n':0});item['correct']+=hit;item['n']+=1
    if not torch.isfinite(state).all():raise FloatingPointError('state became nonfinite')
    return torch.stack(losses).mean(),{'correct':correct,'n':total,'accuracy':correct/total,'by_write_age':by_age}


def run(seed,args):
    torch.manual_seed(seed);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=RecurrentWorkingState(width=args.width).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    generator=torch.Generator().manual_seed(seed+111000);start=time.perf_counter();curve=[]
    for update in range(args.updates):
        loss,metric=stream(model,generator,32,24,device,train=True)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if (update+1)%100==0:curve.append({'update':update+1,'loss':float(loss.detach()),'accuracy':metric['accuracy']})
    model.eval();before=weights_hash(model);evaluation={}
    with torch.no_grad():
        for length in (24,96,384):
            for mode in ('state','query_reset'):
                gen=torch.Generator().manual_seed(seed+112000+length)
                _,result=stream(model,gen,64,length,device,reset_query=mode=='query_reset')
                evaluation[f'{length}/{mode}']=result
    if before!=weights_hash(model):raise AssertionError('evaluation changed parameters')
    return {'seed':seed,'seconds':time.perf_counter()-start,'curve':curve,'evaluation':evaluation,
            'state_floats_per_stream':args.width,'state_bytes_fp32':args.width*4,
            'model_parameters':sum(p.numel() for p in model.parameters()),'weights_sha256':before,
            'raw_history_access_at_query':False,'eval_parameters_unchanged':True,
            'limitations':['symbolic writes/queries','8 known slots/classes','training BPTT activations not constant memory',
                          'evaluation truth/metrics stored externally but never query inputs','not PBSLM main model integration']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--updates',type=int,default=600);p.add_argument('--width',type=int,default=96)
    p.add_argument('--output',default='experiments/recurrent_state/pilot600.json');args=p.parse_args()
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    if min(args.updates,args.width)<1:p.error('positive budgets required')
    report={'config':vars(args),'scope':'learned fixed recurrent state, no raw history query, symbolic mechanism only',
            'provenance':capture(['models/recurrent_working_state.py','evaluation/recurrent_state_control.py']),'runs':[]}
    for seed in args.seeds:
        result=run(seed,args);report['runs'].append(result)
        print(json.dumps({'seed':seed,'seconds':result['seconds'],'accuracy':{k:v['accuracy'] for k,v in result['evaluation'].items()}}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
