"""Fresh continuous identities, overwrite and unknown queries without ID access.

Fresh random keys every episode. Evaluator uses index/seen truth only to construct
query/target; learned model gets only write vector/value and query vector.
A strong key-value bank reference stores all8 keys/latest observed values/seen bits
and cosine threshold. This is explicit algorithmic storage, not learned model.
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


def state_hash(model):
    h=hashlib.sha256()
    for n,v in model.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def episode(model,gen,batch,steps,device,*,training=False,noise=0.,reset=False):
    keys=F.normalize(torch.randn(batch,8,16,generator=gen),dim=-1).to(device)
    h=model.initial(batch);truth=torch.full((batch,8),8,device=device,dtype=torch.long)
    seen=torch.zeros(batch,8,dtype=torch.bool,device=device);rows=torch.arange(batch,device=device)
    losses=[];counts={group:{'n':0,'correct':0} for group in ('known','unknown')};baseline={group:{'n':0,'correct':0} for group in counts}
    for step in range(steps):
        index=torch.randint(8,(batch,),generator=gen).to(device);value=torch.randint(8,(batch,),generator=gen).to(device)
        h=model.observe(h,keys[rows,index],value);truth[rows,index]=value;seen[rows,index]=True
        # Known branch uniform over all8 keys includes unobserved identities, which
        # must also yield unknown; other branch is fresh absent key, not old-ID hint.
        qindex=torch.randint(8,(batch,),generator=gen).to(device)
        fresh=F.normalize(torch.randn(batch,16,generator=gen),dim=-1).to(device)
        absent=(torch.rand(batch,generator=gen)<.25).to(device)
        query=torch.where(absent[:,None],fresh,keys[rows,qindex])
        eps=torch.randn(batch,16,generator=gen).to(device)
        query=F.normalize(query+noise*eps,dim=-1)
        target=torch.where(absent,torch.full_like(qindex,8),truth[rows,qindex])
        output=model.query(torch.zeros_like(h) if reset else h,query)
        losses.append(F.cross_entropy(output,target))
        if not training:
            predicted=output.argmax(-1)
            sim=torch.einsum('bd,bkd->bk',query,keys);sim=sim.masked_fill(~seen,-2)
            top=sim.argmax(-1);score=sim.max(-1).values
            reference=torch.where(score>=.9,truth[rows,top],torch.full_like(top,8))
            for group,mask in [('known',target!=8),('unknown',target==8)]:
                counts[group]['n']+=int(mask.sum());counts[group]['correct']+=int(((predicted==target)&mask).sum())
                baseline[group]['n']+=int(mask.sum());baseline[group]['correct']+=int(((reference==target)&mask).sum())
    for metrics in (counts,baseline):
        for group in metrics:
            m=metrics[group];m['accuracy']=m['correct']/m['n'] if m['n'] else None
    return torch.stack(losses).mean(),{'learned':counts,'key_value_bank_reference':baseline,
              'state_finite':bool(torch.isfinite(h).all())}


def run(seed,args):
    torch.manual_seed(seed);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=ContinuousKeyState().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.002)
    gen=torch.Generator().manual_seed(seed+151000);curve=[];start=time.perf_counter()
    for update in range(args.updates):
        loss,_=episode(model,gen,32,24,device,training=True)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if (update+1)%100==0:curve.append({'update':update+1,'loss':float(loss.detach())})
    model.eval();before=state_hash(model);evaluations={}
    with torch.no_grad():
        for length in (24,96):
            for noise in (0.,.05):
                for reset in (False,True):
                    gen=torch.Generator().manual_seed(seed+152000+length)
                    _,result=episode(model,gen,64,length,device,noise=noise,reset=reset)
                    if not result['state_finite']:raise AssertionError('nonfinite working state')
                    evaluations[f'{length}/noise{noise}/reset{reset}']=result
    if before!=state_hash(model):raise AssertionError('eval changed weights')
    path=ROOT/args.output_dir/f'seed{seed}.pt'
    torch.save({'format':'pbslm.continuous_key_state.research.v1','config':{'key_dim':16,'values':8,'width':128},
                'seed':seed,'updates':args.updates,'state_dict':model.cpu().state_dict()},path)
    return {'seed':seed,'curve':curve,'evaluation':evaluations,'weights_sha256':before,
            'checkpoint_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'parameters':sum(p.numel() for p in model.parameters()),'state_fp32_bytes_per_stream':512,
            'reference_key_value_seen_payload_bytes_per_stream':8*16*4+8*8+8,
            'seconds_training_and_eval':time.perf_counter()-start,'eval_weights_unchanged':True,
            'note':'512byte h vs 584byte reference payload excludes parameters/temporary arrays/Python; no fair compute claim'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=600)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--output-dir',default='experiments/recurrent_state/continuous_keys600')
    args=p.parse_args();out=ROOT/args.output_dir
    if out.exists():raise FileExistsError(out)
    if args.updates<1:p.error('positive update budget required')
    out.mkdir(parents=True)
    report={'config':vars(args),'scope':'fresh random continuous-key symbolic associations, not language or PBSLM integration',
            'reference_threshold':.9,'provenance':capture(['models/continuous_key_state.py','evaluation/continuous_key_control.py']), 'runs':[]}
    for seed in args.seeds:
        result=run(seed,args);report['runs'].append(result)
        print(json.dumps({'seed':seed,'eval96':result['evaluation']['96/noise0.0/resetFalse']}),flush=True)
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
