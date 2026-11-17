"""Pair delta/additive fast states on EXACT original fresh-key benchmark streams.

Same128floats as globalGRU, not equal parameters/FLOPs/inductive bias. Value vectors
initialize to onehot identity, read sees aggregate retrieval only. No raw history.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from models.content_fast_state import ContentFastState
from evaluation.continuous_key_control import episode,state_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


def run(seed,rule,args):
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.manual_seed(seed)
    model=ContentFastState(rule=rule).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.002)
    gen=torch.Generator().manual_seed(seed+151000);curve=[];start=time.perf_counter()
    for update in range(args.updates):
        loss,_=episode(model,gen,32,24,device,training=True)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if (update+1)%100==0:curve.append({'update':update+1,'loss':float(loss.detach())})
    model.eval();before=state_hash(model);evaluation={}
    with torch.no_grad():
        for length in (24,96):
            for noise in (0.,.05):
                for reset in (False,True):
                    gen=torch.Generator().manual_seed(seed+152000+length)
                    _,metric=episode(model,gen,64,length,device,noise=noise,reset=reset)
                    if not metric['state_finite']:raise AssertionError('nonfinite state')
                    evaluation[f'{length}/noise{noise}/reset{reset}']=metric
    if before!=state_hash(model):raise AssertionError('evaluation changed weights')
    path=ROOT/args.output_dir/f'{rule}_seed{seed}.pt'
    torch.save({'format':'pbslm.content_fast_state.research.v1','seed':seed,'rule':rule,
                 'config':{'key_dim':16,'values':8},'state_dict':model.cpu().state_dict()},path)
    return {'seed':seed,'rule':rule,'evaluation':evaluation,'curve':curve,
            'state_floats':128,'state_bytes_fp32':512,'parameters':sum(p.numel() for p in model.parameters()),
            'write_rate':float(model.write_logit.sigmoid().detach()),'weights_sha256':before,
            'checkpoint_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'seconds_training_and_eval':time.perf_counter()-start,'eval_weights_unchanged':True,
            'note':'onehot value initialization and explicit associative operation are extra priors; no compute-saving or language claim'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=600)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--output-dir',default='experiments/recurrent_state/content_fast600')
    args=p.parse_args();out=ROOT/args.output_dir
    if out.exists():raise FileExistsError(out)
    if args.updates<1:p.error('positive update budget required')
    out.mkdir(parents=True)
    report={'config':vars(args),'scope':'fresh-key associative state versus additive, offline fixed-weight evaluation',
            'provenance':capture(['models/content_fast_state.py','evaluation/content_fast_control.py',
                                 'evaluation/continuous_key_control.py']), 'runs':[]}
    for seed in args.seeds:
        for rule in ('delta','additive'):
            result=run(seed,rule,args);report['runs'].append(result)
            print(json.dumps({'seed':seed,'rule':rule,'eval96':result['evaluation']['96/noise0.0/resetFalse']}),flush=True)
            (out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
