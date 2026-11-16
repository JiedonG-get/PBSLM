"""Discriminate M-assisted retrieval from M-free A consolidation.

Tiny existing compact-BPE checkpoint; mechanism screen, not language benchmark.
Teacher frozen with M; student M disabled and zero throughout. O1 six-group updates.
Old facts first trained by labels; new facts either labels or teacher predictions.
No live-runtime optimizer/parameter swap: isolated research copies only.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from evaluation.text_checkpoint import load
from evaluation.text_facts import fill
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


def logits(model,x):
    return F.normalize(model(x,1,sample_noise=False)['z_a'][:,0].float(),dim=-1)@model.code.float().t()/.1


def questions(task,entities,templates):
    return torch.stack([task.inputs[e,t] for e in entities for t in templates])


def protected_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def measure(model,task,old_y,new_y):
    result={}
    for name,entities,labels in [('old',list(range(4)),old_y),('new',list(range(4,8)),new_y)]:
        for split,templates in [('train_surface',(2,3)),('held_surface',(4,5))]:
            x=questions(task,entities,templates)
            prediction=logits(model,x).argmax(-1)
            expected=labels.repeat_interleave(len(templates))
            result[f'{name}/{split}']={'accuracy':float((prediction==expected).float().mean()),
                                      'predicted_ids':prediction.tolist(),'target_ids':expected.tolist()}
    return result


def train_steps(model,task,entities,target_ids,steps,seed,*,replay=None):
    if model.use_mem or any(mem.count_nonzero() for mem in model.mem):
        raise ValueError('consolidation student must have memory disabled and zero')
    parameters=model.online_parameters()
    optimizer=torch.optim.AdamW(parameters,lr=.002)
    rng=torch.Generator().manual_seed(seed);replay_rng=torch.Generator().manual_seed(seed+1);curve=[]
    for step in range(steps):
        # Same current sample/template sequence across direct/teacher arms.
        item=int(torch.randint(len(entities),(),generator=rng));template=int(torch.randint(2,4,(),generator=rng))
        x=[task.inputs[entities[item],template]];y=[int(target_ids[item,template-2])]
        if replay is not None:
            old_i=int(torch.randint(4,(),generator=replay_rng))
            x.append(task.inputs[old_i,template]);y.append(int(replay[old_i]))
        inputs=torch.stack(x);target=torch.tensor(y,device=inputs.device)
        loss=F.cross_entropy(logits(model,inputs),target)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
        if (step+1)%50==0:curve.append({'step':step+1,'loss':float(loss.detach())})
    return curve


def run(seed,args):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,task,artifact=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt',device)
    torch.manual_seed(seed+51000)
    order=torch.randperm(8,device=device)
    old_y=task.labels[order[:4]];new_y=task.labels[order[4:]]
    model.use_mem=False;model.freeze_for_online()
    teacher=copy.deepcopy(model);teacher.use_mem=True
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    protected=protected_hash(model)
    old_curve=train_steps(model,task,list(range(4)),old_y[:,None].expand(-1,2),args.base_steps,seed+52000)
    initial=measure(model,task,old_y,new_y)
    support=questions(task,list(range(4,8)),(0,))
    query=questions(task,list(range(4,8)),(2,3))
    teachers={};teacher_metrics={}
    for mode in ('normal','wrong'):
        fill(teacher,support,new_y,mode)
        with torch.no_grad():pred=logits(teacher,query).argmax(-1).reshape(4,2)
        teachers[mode]=pred
        expected=new_y[:,None].expand(-1,2)
        teacher_metrics[mode]={'accuracy_vs_truth':float((pred==expected).float().mean()),'predicted_ids':pred.tolist()}
    arms={};start=time.perf_counter()
    for name in ('none','direct','teacher','teacher_replay','wrong_teacher'):
        student=copy.deepcopy(model);student.use_mem=False
        before=protected_hash(student)
        curve=[]
        if name!='none':
            targets=new_y[:,None].expand(-1,2) if name=='direct' else teachers['wrong' if name=='wrong_teacher' else 'normal']
            curve=train_steps(student,task,list(range(4,8)),targets,args.steps,seed+53000,
                              replay=old_y if name=='teacher_replay' else None)
        if before!=protected_hash(student) or protected!=protected_hash(student):
            raise AssertionError('frozen/C/shared parameters changed')
        if any(mem.count_nonzero() for mem in student.mem):raise AssertionError('student M not zero')
        arms[name]={'evaluation':measure(student,task,old_y,new_y),'curve':curve,
                    'student_memory_disabled':True,'student_M_zero':True,'protected_parameters_unchanged':True,
                    'optimizer_steps':0 if name=='none' else args.steps,
                    'training_examples':0 if name=='none' else args.steps*(2 if name=='teacher_replay' else 1)}
    return {'seed':seed,'checkpoint':artifact,'base_curve':old_curve,'before_consolidation':initial,
            'old_labels':old_y.tolist(),'new_labels':new_y.tolist(),'teacher':teacher_metrics,
            'arms':arms,'adaptation_seconds':time.perf_counter()-start,
            'warning':'held_surface is narrow unseen template for same entities; replay has 2x training examples, not equal FLOPs'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[0])
    p.add_argument('--base-steps',type=int,default=200);p.add_argument('--steps',type=int,default=200)
    p.add_argument('--output',default='experiments/consolidation/pilot200.json');args=p.parse_args()
    if min(args.base_steps,args.steps)<1:p.error('positive step budgets required')
    path=ROOT/args.output
    if path.exists():raise FileExistsError(path)
    report={'config':vars(args),'scope':'M→A teacher hard-label consolidation mechanism, no general language claim',
            'provenance':capture(['evaluation/memory_consolidation.py','evaluation/text_checkpoint.py','evaluation/text_facts.py',
                                  'models/model.py','models/model_v2.py']), 'runs':[]}
    for seed in args.seeds:
        result=run(seed,args);report['runs'].append(result)
        print(json.dumps({'seed':seed,'teacher':result['teacher'],
                          'before':{k:v['accuracy'] for k,v in result['before_consolidation'].items()},
                          'after':{arm:{k:v['accuracy'] for k,v in data['evaluation'].items()} for arm,data in result['arms'].items()}}),flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
