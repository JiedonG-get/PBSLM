"""Compare frozen E keys and token mean with matched readout size and targets.

Reuses independently saved shared-soft teacher probabilities and labels, verifies
checkpoint identity. Same steps/sampling as O1 soft control, not equal FLOPs or
initial-state equivalence: each feature model learns old facts in its own readout.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from evaluation.text_checkpoint import load
from evaluation.text_facts import frozen_hash
from evaluation.memory_consolidation import train_steps,measure,protected_hash
from evaluation.consolidation_soft_control import soft_steps
from evaluation.provenance import capture
from models.feature_readout import FrozenFeatureReadout

ROOT=Path(__file__).resolve().parents[1]


def run(seed,args,reference):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    old_y=torch.tensor(reference['old_ids'],device=device);new_y=torch.tensor(reference['new_ids'],device=device)
    teacher=torch.tensor(reference['teacher']['normal']['probabilities'],device=device)
    arms={}
    for feature in ('e_key','token_mean'):
        base,task,metadata=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt',device)
        if metadata['sha256']!=reference['checkpoint']['sha256']:raise AssertionError('teacher checkpoint differs')
        if frozen_hash(base)!=reference['teacher']['normal']['non_memory_sha256']:
            raise AssertionError('teacher key representation differs')
        base.use_mem=False
        torch.manual_seed(seed+64000)
        model=FrozenFeatureReadout(base,feature)
        protected=protected_hash(model);start=time.perf_counter()
        curve_old=train_steps(model,task,list(range(4)),old_y[:,None].expand(-1,2),args.base_steps,seed+62000)
        before=measure(model,task,old_y,new_y)
        curve_new=soft_steps(model,task,teacher,args.steps,seed+63000)
        after=measure(model,task,old_y,new_y)
        if protected_hash(model)!=protected:raise AssertionError('frozen base changed')
        if any(m.count_nonzero() for m in model.mem):raise AssertionError('M changed')
        arms[feature]={'before':before,'after':after,'base_curve':curve_old,'new_curve':curve_new,
                       'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                       'frozen_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad),
                       'seconds':time.perf_counter()-start,'frozen_base_unchanged':True,'M_disabled_zero':True}
    if arms['e_key']['trainable_parameters']!=arms['token_mean']['trainable_parameters']:
        raise AssertionError('readout parameter budgets differ')
    return {'seed':seed,'old_ids':old_y.tolist(),'new_ids':new_y.tolist(),'arms':arms,
            'O1_soft_reference':{'before':reference['before'],'after':reference['arms']['soft_teacher']['evaluation']}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--base-steps',type=int,default=200);p.add_argument('--steps',type=int,default=200)
    p.add_argument('--reference',default='experiments/consolidation/shared_soft200.json')
    p.add_argument('--output',default='experiments/consolidation/feature_readout200.json');args=p.parse_args()
    if min(args.base_steps,args.steps)<1:p.error('positive budgets required')
    path=ROOT/args.output
    if path.exists():raise FileExistsError(path)
    ref=json.loads((ROOT/args.reference).read_text(encoding='utf-8'))
    if (args.steps,args.base_steps)!=(ref['config']['steps'],ref['config']['base_steps']):raise ValueError('reference budgets differ')
    report={'config':vars(args),'scope':'isolated learned readout, not full language/Runtime architecture solution',
            'provenance':capture(['models/feature_readout.py','evaluation/feature_readout_control.py',
                                  'evaluation/consolidation_soft_control.py','evaluation/memory_consolidation.py',
                                  'models/model.py','models/model_v2.py']), 'runs':[]}
    for seed in args.seeds:
        reference=next(r for r in ref['runs'] if r['seed']==seed)
        result=run(seed,args,reference);report['runs'].append(result)
        print(json.dumps({'seed':seed,'arms':{k:{'before':{n:d['accuracy'] for n,d in v['before'].items()},
                  'after':{n:d['accuracy'] for n,d in v['after'].items()},'params':v['trainable_parameters']} for k,v in result['arms'].items()}}),flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
