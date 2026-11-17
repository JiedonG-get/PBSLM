"""Paired fixed-state global vs learned/explicit slots, no inference history access.

Same96float activation payload; NOT matched parameter count/FLOPs. Symbolic8ID task,
explicit addressing is an oracle baseline, learned routing still receives entity ID.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from models.recurrent_working_state import RecurrentWorkingState
from models.slotted_working_state import SlottedWorkingState
from evaluation.recurrent_state_control import stream,weights_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


def run(seed,kind,args):
    torch.manual_seed(seed);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=(RecurrentWorkingState(width=96) if kind=='global' else SlottedWorkingState(addressing=kind)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    gen=torch.Generator().manual_seed(seed+111000);curve=[];start=time.perf_counter()
    for update in range(args.updates):
        loss,metric=stream(model,gen,32,24,device,train=True)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if (update+1)%100==0:curve.append({'update':update+1,'loss':float(loss.detach()),'accuracy':metric['accuracy']})
    model.eval();before=weights_hash(model);evaluation={}
    with torch.no_grad():
        for length in (24,96,384):
            for mode in ('state','query_reset'):
                gen=torch.Generator().manual_seed(seed+112000+length)
                _,metric=stream(model,gen,64,length,device,reset_query=mode=='query_reset')
                evaluation[f'{length}/{mode}']=metric
    if before!=weights_hash(model):raise AssertionError('evaluation changed parameters')
    out=ROOT/args.output_dir/f'{kind}_seed{seed}.pt'
    torch.save({'format':'pbslm.symbolic_working_state.v1','architecture':kind,'seed':seed,
                'state_dict':{n:v.detach().cpu() for n,v in model.state_dict().items()},
                'config':{'entities':8,'values':8,'state_floats':96},'updates':args.updates},out)
    route=None
    if kind!='global':
        with torch.no_grad():p=model.routing(torch.arange(8,device=device))
        route={'weights':p.cpu().tolist(),'top_slots':p.argmax(-1).tolist(),
               'distinct_top_slots':len(p.argmax(-1).unique()),
               'mean_entropy':float(-(p*p.clamp_min(1e-12).log()).sum(-1).mean())}
    return {'seed':seed,'architecture':kind,'seconds_training_and_eval':time.perf_counter()-start,
            'curve':curve,'evaluation':evaluation,'routing':route,'activation_floats':96,'activation_bytes_fp32':384,
            'parameters':sum(p.numel() for p in model.parameters()),'weights_sha256':before,
            'checkpoint':str(out),'eval_parameters_unchanged':True,'raw_history_query_access':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--updates',type=int,default=600);p.add_argument('--output-dir',default='experiments/recurrent_state/slots600')
    args=p.parse_args();out=ROOT/args.output_dir
    if out.exists():raise FileExistsError(out)
    if args.updates<1:p.error('positive updates required')
    out.mkdir(parents=True)
    report={'config':vars(args),'scope':'paired symbolicID working-state mechanism,96floats not matched whole resources',
            'provenance':capture(['models/recurrent_working_state.py','models/slotted_working_state.py',
                                  'evaluation/recurrent_state_control.py','evaluation/slotted_state_control.py']), 'runs':[]}
    for seed in args.seeds:
        arms={}
        for kind in ('global','learned','explicit'):
            result=run(seed,kind,args);arms[kind]=result
            print(json.dumps({'seed':seed,'architecture':kind,'accuracy':{k:v['accuracy'] for k,v in result['evaluation'].items()},
                              'routing':result['routing']}),flush=True)
        report['runs'].append({'seed':seed,'arms':arms})
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
