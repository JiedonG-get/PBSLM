"""Fixed-budget sparse-write training controls, uniform and all-hot evaluation.

λ=.1 prior fixed before running. Not tuned on test. All symbol IDs seen in training.
No raw history queries. Runtime/model integration outside this mechanism experiment.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from models.slotted_working_state import SlottedWorkingState
from models.sparse_working_state import SparseWorkingState
from evaluation.recurrent_state_control import stream,weights_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def all_hot(model,seed,device,steps=2048,batch=16):
    cases=[]
    for hot in range(8):
        gen=torch.Generator().manual_seed(seed+131000+hot*100)
        truth=torch.randint(8,(batch,8),generator=gen).to(device);state=model.initial(batch)
        for e in range(8):state=model.observe(state,torch.full((batch,),e,device=device,dtype=torch.long),truth[:,e])
        def probe():
            prediction=torch.stack([model.query(state,torch.full((batch,),e,device=device,dtype=torch.long)).argmax(-1) for e in range(8)],1)
            return {'predicted':prediction.cpu().tolist(),'targets':truth.cpu().tolist(),
                    'correct_per_entity':(prediction==truth).sum(0).cpu().tolist(),'n_per_entity':batch}
        initial=probe()
        for _ in range(steps):
            values=torch.randint(8,(batch,),generator=gen).to(device);truth[:,hot]=values
            state=model.observe(state,torch.full((batch,),hot,device=device,dtype=torch.long),values)
        cases.append({'hot':hot,'steps':steps,'initial':initial,'final':probe()})
    return cases


def run(seed,arm,args):
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.manual_seed(seed)
    model=(SlottedWorkingState() if arm=='soft' else SparseWorkingState()).to(device)
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    gen=torch.Generator().manual_seed(seed+111000);curve=[];start=time.perf_counter()
    for update in range(args.updates):
        loss,metric=stream(model,gen,32,24,device,train=True)
        prior=model.collision_loss() if arm=='sparse_collision' else loss.new_zeros(())
        objective=loss+.1*prior
        optimizer.zero_grad(set_to_none=True);objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if (update+1)%100==0:curve.append({'update':update+1,'CE':float(loss.detach()),'collision':float(prior.detach()),'accuracy':metric['accuracy']})
    model.eval();before=weights_hash(model);uniform={}
    with torch.no_grad():
        for length in (24,96,384):
            gen=torch.Generator().manual_seed(seed+112000+length)
            _,metric=stream(model,gen,64,length,device)
            uniform[str(length)]=metric
        cases=all_hot(model,seed,device)
        route=model.routing(torch.arange(8,device=device))
    if weights_hash(model)!=before:raise AssertionError('eval parameter change')
    path=ROOT/args.output_dir/f'{arm}_seed{seed}.pt'
    torch.save({'format':'pbslm.sparse_state.research.v1','arm':arm,'seed':seed,'updates':args.updates,
                'collision_coefficient':.1 if arm=='sparse_collision' else 0.,'state_dict':model.cpu().state_dict()},path)
    hot=[];cold=[]
    for c in cases:
        counts=c['final']['correct_per_entity'];n=c['final']['n_per_entity'];h=c['hot']
        hot.append(counts[h]/n);cold.append(sum(counts[e] for e in range(8) if e!=h)/(7*n))
    result={'seed':seed,'arm':arm,'uniform':uniform,'all_hot':cases,'curve':curve,
            'route':route.cpu().tolist(),'top_slots':route.argmax(-1).cpu().tolist(),
            'distinct_top_slots':len(route.argmax(-1).unique()),'weights_sha256':before,
            'state_floats':96,'parameters':sum(p.numel() for p in model.parameters()),
            'eval_weights_unchanged':True,'seconds_training_and_eval':time.perf_counter()-start}
    print(json.dumps({'seed':seed,'arm':arm,'uniform384':uniform['384']['accuracy'],
                     'all_hot_mean':sum(hot)/8,'all_cold_mean':sum(cold)/8,'distinct_top_slots':result['distinct_top_slots']}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=600)
    p.add_argument('--output-dir',default='experiments/recurrent_state/sparse_train600');args=p.parse_args()
    out=ROOT/args.output_dir
    if out.exists():raise FileExistsError(out)
    if args.updates<1:p.error('positive updates required')
    out.mkdir(parents=True)
    report={'config':vars(args),'scope':'same-data96float known-ID sparse training, soft read, ST gradient, fixed .1 prior',
            'provenance':capture(['models/sparse_working_state.py','models/slotted_working_state.py',
                                 'evaluation/recurrent_state_control.py','evaluation/train_sparse_state.py']), 'runs':[]}
    for seed in (0,1,2):
        for arm in ('soft','sparse','sparse_collision'):
            report['runs'].append(run(seed,arm,args))
            (out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
