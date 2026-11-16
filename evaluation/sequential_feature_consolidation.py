"""Three-stage consolidation and fresh-random-E ablation of frozen readout.

Randomization replaces ONLY ebank and e_mix in a checkpoint copy; pretrained tok/
code remain shared. Teacher is independent original checkpoint and never randomized.
Three groups use familiar entities0..11 and same four colors. No Runtime integration.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from models.model_v2 import PBSLMv2
from models.feature_readout import FrozenFeatureReadout
from evaluation.text_checkpoint import load
from evaluation.text_facts import model_config,fill,frozen_hash
from evaluation.memory_consolidation import logits,questions,protected_hash
from evaluation.consolidation_soft_control import distillation_loss,memory_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


def randomize_e_only(base,task,seed):
    before={name:value.detach().clone() for name,value in base.state_dict().items()
            if name!='e_mix' and not name.startswith('ebank.')}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fresh=PBSLMv2(**model_config(task)).eval()
    base.ebank.load_state_dict(fresh.ebank.state_dict(),strict=True)
    with torch.no_grad():base.e_mix.copy_(fresh.e_mix.to(base.e_mix.device))
    for name,value in before.items():
        if not torch.equal(value,base.state_dict()[name]):raise AssertionError('random E changed unrelated state')
    return {'random_seed':seed,'operation':'fresh initialization of ebank/e_mix only; embedding/code and all other state unchanged'}


def train_stage(model,task,entities,labels,probabilities,stage,seed,steps):
    if model.use_mem or any(m.count_nonzero() for m in model.mem):raise ValueError('student M must remain off/zero')
    optimizer=torch.optim.AdamW(model.online_parameters(),lr=.002)
    rng=torch.Generator().manual_seed(seed+72000+stage);curve=[]
    for step in range(steps):
        i=int(torch.randint(4,(),generator=rng));t=int(torch.randint(2,4,(),generator=rng))
        scores=logits(model,task.inputs[entities[i],t][None])
        loss=(F.cross_entropy(scores,labels[i:i+1]) if stage==0 else
              distillation_loss(scores,probabilities[i,t-2][None]))
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.online_parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if (step+1)%50==0:curve.append({'step':step+1,'loss':float(loss.detach())})
    return curve


@torch.no_grad()
def evaluate(model,task,groups,labels):
    metrics={}
    for g,(entities,y) in enumerate(zip(groups,labels)):
        for name,templates in [('train',(2,3)),('held',(4,5))]:
            prediction=logits(model,questions(task,entities,templates)).argmax(-1)
            targets=y.repeat_interleave(2)
            metrics[f'group{g}/{name}']={'accuracy':float((prediction==targets).float().mean()),
                                      'predicted':prediction.tolist(),'targets':targets.tolist()}
    return metrics


def run(seed,args):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    base,task,artifact=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt',device)
    base.use_mem=False
    groups=[list(range(i*4,(i+1)*4)) for i in range(3)]
    rng=torch.Generator().manual_seed(seed+71000)
    colors=torch.randperm(8,generator=rng)[:4].to(device)
    labels=[task.labels[colors[torch.randperm(4,generator=rng).to(device)]] for _ in groups]
    teacher=copy.deepcopy(base);teacher.use_mem=True
    for p in teacher.parameters():p.requires_grad_(False)
    teacher_nonM=frozen_hash(teacher);teacher_data=[];probabilities=[]
    for entities,y in zip(groups,labels):
        fill(teacher,questions(task,entities,(0,)),y,'normal');mhash=memory_hash(teacher)
        with torch.no_grad():prob=logits(teacher,questions(task,entities,(2,3))).softmax(-1).reshape(4,2,-1).detach().clone()
        if mhash!=memory_hash(teacher) or teacher_nonM!=frozen_hash(teacher):raise AssertionError('teacher mutated')
        probabilities.append(prob)
        teacher_data.append({'truth_top1':float((prob.argmax(-1)==y[:,None]).float().mean()),
                             'probabilities':prob.cpu().tolist(),'memory_sha256':mhash})
    arms={}
    for arm in ('trained_e','random_e','token_mean'):
        candidate=copy.deepcopy(base)
        mutation=randomize_e_only(candidate,task,seed+73000) if arm=='random_e' else None
        torch.manual_seed(seed+74000)
        student=FrozenFeatureReadout(candidate,'token_mean' if arm=='token_mean' else 'e_key')
        protected=protected_hash(student);start=time.perf_counter()
        stages=[];initial=evaluate(student,task,groups,labels)
        for stage,(entities,y,prob) in enumerate(zip(groups,labels,probabilities)):
            curve=train_stage(student,task,entities,y,prob,stage,seed,args.steps)
            if protected!=protected_hash(student):raise AssertionError('base changed during readout training')
            stages.append({'stage':stage,'curve':curve,'evaluation':evaluate(student,task,groups,labels)})
        arms[arm]={'mutation':mutation,'initial':initial,'stages':stages,
                   'seconds':time.perf_counter()-start,'trainable_parameters':sum(p.numel() for p in student.online_parameters()),
                   'frozen_base_unchanged':True,'student_M_zero':all(not m.count_nonzero() for m in student.mem)}
    return {'seed':seed,'checkpoint':artifact,'groups':groups,'labels':[y.tolist() for y in labels],
            'teacher_non_memory_sha256':teacher_nonM,'teacher':teacher_data,'arms':arms}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--steps',type=int,default=200);p.add_argument('--output',default='experiments/consolidation/sequence_random_e200.json')
    args=p.parse_args();out=ROOT/args.output
    if args.steps<1:p.error('positive steps required')
    if out.exists():raise FileExistsError(out)
    report={'config':vars(args),'scope':'three groups familiar entities, frozen readout; fresh E ablates all E learning not alignment alone',
            'provenance':capture(['evaluation/sequential_feature_consolidation.py','models/feature_readout.py',
                                  'evaluation/consolidation_soft_control.py','models/model.py','models/model_v2.py']), 'runs':[]}
    for seed in args.seeds:
        result=run(seed,args);report['runs'].append(result)
        print(json.dumps({'seed':seed,'teacher':[t['truth_top1'] for t in result['teacher']],
                          'final':{a:{k:v['accuracy'] for k,v in d['stages'][-1]['evaluation'].items()}
                                   for a,d in result['arms'].items()}}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
