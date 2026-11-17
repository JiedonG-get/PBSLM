"""Continuous random-key state, corrections, unknowns and dynamic exact-key baseline.

Fresh 8 normalized 16D keys and random 8-value permutation per episode. After initial
writes, events may correct a key then query; unknown keys are 25%. No key index,
seen mask, raw history or language tokens enter the recurrent model. This is an
associative mechanism task, NOT natural-language understanding or PBSLM integration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from models.continuous_key_state import ContinuousKeyState
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]
UNKNOWN=8


def rand_unit(shape,generator,device):
    return F.normalize(torch.randn(shape,generator=generator).to(device),dim=-1)


def initialize_episode(model,batch,generator,device):
    state=model.initial(batch);bank=rand_unit((batch,8,16),generator,device)
    values=torch.stack([torch.randperm(8,generator=generator) for _ in range(batch)]).to(device)
    order=torch.stack([torch.randperm(8,generator=generator) for _ in range(batch)]).to(device)
    for pos in range(8):
        ids=order[:,pos]
        state=model.observe(state,bank[torch.arange(batch,device=device),ids],values[torch.arange(batch,device=device),ids])
    return state,bank,values


def train_episode(model,batch,total_events,generator,device):
    if total_events<9:raise ValueError('need eight initial writes and at least one event')
    state,bank,values=initialize_episode(model,batch,generator,device);losses=[]
    for _ in range(8,total_events):
        correction=(torch.rand(batch,generator=generator)<.25).to(device)
        ids=torch.randint(8,(batch,),generator=generator).to(device)
        new_values=torch.randint(8,(batch,),generator=generator).to(device)
        candidate=model.observe(state,bank[torch.arange(batch,device=device),ids],new_values)
        state=torch.where(correction[:,None],candidate,state)
        values[torch.arange(batch,device=device),ids]=torch.where(correction,new_values,values[torch.arange(batch,device=device),ids])
        unknown=(torch.rand(batch,generator=generator)<.25).to(device)
        query_id=torch.randint(8,(batch,),generator=generator).to(device)
        query=torch.where(unknown[:,None],rand_unit((batch,16),generator,device),bank[torch.arange(batch,device=device),query_id])
        sigma=float(torch.rand((),generator=generator))*.05
        query=F.normalize(query+sigma*torch.randn(query.shape,generator=generator).to(device),dim=-1)
        target=torch.where(unknown,torch.full((batch,),UNKNOWN,device=device),values[torch.arange(batch,device=device),query_id])
        losses.append(F.cross_entropy(model.query(state,query),target))
    return torch.stack(losses).mean()


def combine_state_fallback(state_prediction,dictionary_prediction,*,unknown_id=UNKNOWN):
    if state_prediction.shape!=dictionary_prediction.shape:
        raise ValueError('state and fallback batch shapes differ')
    return torch.where(state_prediction==unknown_id,dictionary_prediction,state_prediction)


def evaluate_episode(model,batch,total_events,generator,device,key_noise,dictionary_threshold):
    state,bank,values=initialize_episode(model,batch,generator,device)
    counters={name:{'correct':0,'known_correct':0,'known_n':0,'unknown_correct':0,'unknown_n':0,
                    'false_accept':0,'known_coverage':0,'fallback_calls':0,'fallback_correct':0,'total':0}
              for name in ('state','reset','dictionary','state_abstain_fallback')}
    correction_count=0
    for _ in range(8,total_events):
        correction=(torch.rand(batch,generator=generator)<.25).to(device)
        ids=torch.randint(8,(batch,),generator=generator).to(device)
        new_values=torch.randint(8,(batch,),generator=generator).to(device)
        candidate=model.observe(state,bank[torch.arange(batch,device=device),ids],new_values)
        state=torch.where(correction[:,None],candidate,state)
        values[torch.arange(batch,device=device),ids]=torch.where(correction,new_values,values[torch.arange(batch,device=device),ids])
        correction_count+=int(correction.sum())
        unknown=(torch.rand(batch,generator=generator)<.25).to(device)
        query_id=torch.randint(8,(batch,),generator=generator).to(device)
        query=torch.where(unknown[:,None],rand_unit((batch,16),generator,device),bank[torch.arange(batch,device=device),query_id])
        query=F.normalize(query+key_noise*torch.randn(query.shape,generator=generator).to(device),dim=-1)
        target=torch.where(unknown,torch.full((batch,),UNKNOWN,device=device),values[torch.arange(batch,device=device),query_id])
        known=~unknown
        neural=model.query(state,query).argmax(-1)
        reset=model.query(torch.zeros_like(state),query).argmax(-1)
        similarity=torch.einsum('bd,bkd->bk',query,bank)
        best=similarity.argmax(-1);nearest_value=values.gather(1,best[:,None])[:,0]
        dictionary=torch.where(similarity.max(-1).values>=dictionary_threshold,nearest_value,torch.full_like(target,UNKNOWN))
        hybrid=combine_state_fallback(neural,dictionary)
        for name,prediction in [('state',neural),('reset',reset),('dictionary',dictionary),('state_abstain_fallback',hybrid)]:
            c=counters[name];hit=prediction==target
            c['correct']+=int(hit.sum());c['known_correct']+=int(hit[known].sum());c['known_n']+=int(known.sum())
            c['unknown_correct']+=int(hit[unknown].sum());c['unknown_n']+=int(unknown.sum())
            c['false_accept']+=int((prediction[unknown]!=UNKNOWN).sum())
            c['known_coverage']+=int((prediction[known]!=UNKNOWN).sum())
            c['fallback_calls']+=int((neural==UNKNOWN).sum()) if name=='state_abstain_fallback' else 0
            c['fallback_correct']+=int(((neural==UNKNOWN)&(dictionary==target)).sum()) if name=='state_abstain_fallback' else 0
            c['total']+=batch
    result={}
    for name,c in counters.items():
        result[name]={'accuracy':c['correct']/c['total'],'known_accuracy':c['known_correct']/c['known_n'],
            'unknown_accuracy':c['unknown_correct']/c['unknown_n'],
            'unknown_false_accept_rate':c['false_accept']/c['unknown_n'],
            'known_coverage':c['known_coverage']/c['known_n'],
            'fallback_call_rate':c['fallback_calls']/c['total'],'fallback_correct':c['fallback_correct'],**c}
    return {'metrics':result,'total_events':total_events,'queries':batch*(total_events-8),
            'correction_writes':correction_count,'query_key_noise_sigma':key_noise,'memory_keys':8}


def calibrate_dictionary_threshold(seed,args,device):
    generator=torch.Generator().manual_seed(seed+140000);known_sims=[];unknown_sims=[]
    for _ in range(args.validation_batches):
        bank=rand_unit((32,8,16),generator,device)
        ids=torch.randint(8,(32,),generator=generator).to(device)
        known=F.normalize(bank[torch.arange(32,device=device),ids]+.025*torch.randn(32,16,generator=generator).to(device),dim=-1)
        unknown=F.normalize(rand_unit((32,16),generator,device)+.025*torch.randn(32,16,generator=generator).to(device),dim=-1)
        known_sims.extend(torch.einsum('bd,bkd->bk',known,bank).max(-1).values.cpu().tolist())
        unknown_sims.extend(torch.einsum('bd,bkd->bk',unknown,bank).max(-1).values.cpu().tolist())
    rows=[]
    for threshold in [i/100 for i in range(20,96,2)]:
        tpr=sum(x>=threshold for x in known_sims)/len(known_sims)
        fpr=sum(x>=threshold for x in unknown_sims)/len(unknown_sims)
        rows.append({'threshold':threshold,'balanced_accuracy':.5*(tpr+1-fpr),'known_coverage':tpr,'unknown_false_accept_rate':fpr})
    return max(rows,key=lambda row:(row['balanced_accuracy'],row['threshold'])),rows


def run(seed,args):
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.manual_seed(seed)
    model=ContinuousKeyState(width=args.width).to(device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    generator=torch.Generator().manual_seed(seed+130000);curve=[];start=time.perf_counter()
    for update in range(args.updates):
        loss=train_episode(model,32,24,generator,device)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if (update+1)%100==0:curve.append({'update':update+1,'loss':float(loss.detach())})
    model.eval();digest_before=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    threshold,grid=calibrate_dictionary_threshold(seed,args,device);evaluation={}
    with torch.no_grad():
        for length in (24,96):
            for noise in (0.,.05):
                generator=torch.Generator().manual_seed(seed+150000+length*100+int(noise*10000))
                evaluation[f'{length}/{noise}']=evaluate_episode(model,32,length,generator,device,noise,threshold['threshold'])
    digest_after=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    if digest_before!=digest_after:raise AssertionError('evaluation modified model weights')
    return {'seed':seed,'updates':args.updates,'curve':curve,'dictionary_threshold':threshold,
            'threshold_grid':grid,'evaluation':evaluation,'parameters':sum(p.numel() for p in model.parameters()),
            'state_floats':args.width,'state_bytes_fp32':args.width*4,'weights_sha256':digest_after,
            'validation_batches':args.validation_batches,'test_batches_per_event_stream':32,
            'seconds_train_and_eval':time.perf_counter()-start,
            'scope_limit':'8 random keys; training corrections, fixed-dimension key/value classes; symbolic only, not PBSLM'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--updates',type=int,default=800);parser.add_argument('--width',type=int,default=96)
    parser.add_argument('--validation-batches',type=int,default=16)
    parser.add_argument('--output',default='experiments/recurrent_state/continuous_keys_correction800.json')
    args=parser.parse_args();out=ROOT/args.output
    if min(args.updates,args.width,args.validation_batches)<1:parser.error('positive budgets required')
    if out.exists():raise FileExistsError(out)
    report={'config':vars(args),'scope':'random continuous keys with unknown query and correction events; no natural language',
            'provenance':capture(['models/continuous_key_state.py','evaluation/continuous_key_state_control.py']), 'runs':[]}
    for seed in args.seeds:
        row=run(seed,args);report['runs'].append(row)
        print(json.dumps({'seed':seed,'eval':{key:{k:v for k,v in data['metrics'].items()} for key,data in row['evaluation'].items()},
                          'dictionary_threshold':row['dictionary_threshold']}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
