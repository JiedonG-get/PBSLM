"""Shared-answer-vocabulary, soft/zero/wrong M teacher consolidation control.

All targets cached before student adaptation. Original stable teacher, M-free O1
student copies; no Runtime/live model updates. No claim of language generalization.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from evaluation.memory_consolidation import logits,questions,measure,train_steps,protected_hash
from evaluation.text_checkpoint import load
from evaluation.text_facts import fill,frozen_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


def memory_hash(model):
    h=hashlib.sha256()
    for mem in model.mem:h.update(mem.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def distillation_loss(student_logits,teacher_probabilities):
    if student_logits.shape!=teacher_probabilities.shape:raise ValueError('teacher/student shape mismatch')
    if not torch.isfinite(teacher_probabilities).all() or (teacher_probabilities<0).any():
        raise ValueError('finite nonnegative teacher distribution required')
    if not torch.allclose(teacher_probabilities.sum(-1),torch.ones_like(teacher_probabilities.sum(-1)),atol=1e-5):
        raise ValueError('teacher probabilities must sum to one')
    return F.kl_div(student_logits.log_softmax(-1),teacher_probabilities.detach(),reduction='batchmean')


def soft_steps(model,task,probabilities,steps,seed):
    if model.use_mem or any(mem.count_nonzero() for mem in model.mem):raise ValueError('student must be M-free')
    parameters=model.online_parameters();allowed={id(p) for p in parameters}
    if {id(p) for p in model.parameters() if p.requires_grad}!=allowed:raise ValueError('O1 freeze contract required')
    optimizer=torch.optim.AdamW(parameters,lr=.002);rng=torch.Generator().manual_seed(seed);curve=[]
    for step in range(steps):
        item=int(torch.randint(4,(),generator=rng));template=int(torch.randint(2,4,(),generator=rng))
        x=task.inputs[item+4,template][None]
        loss=distillation_loss(logits(model,x),probabilities[item,template-2][None])
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
        if (step+1)%50==0:curve.append({'step':step+1,'KL':float(loss.detach())})
    return curve


def run(seed,args):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,task,artifact=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt',device)
    gen=torch.Generator().manual_seed(seed+61000)
    colors=torch.randperm(8,generator=gen)[:4]
    old_y=task.labels[colors.to(device)]
    new_y=old_y[torch.randperm(4,generator=gen).to(device)]
    model.use_mem=False;model.freeze_for_online()
    teacher=copy.deepcopy(model);teacher.use_mem=True
    for p in teacher.parameters():p.requires_grad_(False)
    teacher_state=frozen_hash(teacher)
    base_curve=train_steps(model,task,list(range(4)),old_y[:,None].expand(-1,2),args.base_steps,seed+62000)
    baseline=measure(model,task,old_y,new_y);protected=protected_hash(model)
    probabilities={};teacher_metrics={}
    sx=questions(task,list(range(4,8)),(0,));qx=questions(task,list(range(4,8)),(2,3))
    for mode in ('normal','zero','wrong'):
        fill(teacher,sx,new_y,mode);before_m=memory_hash(teacher)
        with torch.no_grad():scores=logits(teacher,qx);probs=scores.softmax(-1).reshape(4,2,-1).detach().clone()
        if memory_hash(teacher)!=before_m or frozen_hash(teacher)!=teacher_state:raise AssertionError('teacher forward mutated state')
        probabilities[mode]=probs
        teacher_metrics[mode]={'top1_truth':float((probs.argmax(-1)==new_y[:,None]).float().mean()),
                              'mean_entropy':float(-(probs*probs.clamp_min(1e-12).log()).sum(-1).mean()),
                              'top1_ids':probs.argmax(-1).tolist(),'probabilities':probs.cpu().tolist(),
                              'memory_sha256':before_m,'non_memory_sha256':teacher_state}
    start=time.perf_counter();arms={}
    for arm in ('none','direct','hard_teacher','soft_teacher','zero_soft_teacher','wrong_soft_teacher'):
        student=copy.deepcopy(model);curve=[]
        if arm=='direct':curve=train_steps(student,task,list(range(4,8)),new_y[:,None].expand(-1,2),args.steps,seed+63000)
        elif arm=='hard_teacher':curve=train_steps(student,task,list(range(4,8)),probabilities['normal'].argmax(-1),args.steps,seed+63000)
        elif arm!='none':
            mode={'soft_teacher':'normal','zero_soft_teacher':'zero','wrong_soft_teacher':'wrong'}[arm]
            curve=soft_steps(student,task,probabilities[mode],args.steps,seed+63000)
        if protected_hash(student)!=protected or any(m.count_nonzero() for m in student.mem):raise AssertionError('protected parameters/M changed')
        arms[arm]={'evaluation':measure(student,task,old_y,new_y),'curve':curve,
                   'M_disabled_and_zero':not student.use_mem,'protected_parameters_unchanged':True,
                   'optimizer_steps':0 if arm=='none' else args.steps}
    return {'seed':seed,'checkpoint':artifact,'old_ids':old_y.tolist(),'new_ids':new_y.tolist(),
            'shared_answer_set':set(old_y.tolist())==set(new_y.tolist()),'base_curve':base_curve,
            'before':baseline,'teacher':teacher_metrics,'arms':arms,'adapt_seconds':time.perf_counter()-start}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--base-steps',type=int,default=200);p.add_argument('--steps',type=int,default=200)
    p.add_argument('--output',default='experiments/consolidation/shared_soft200.json');args=p.parse_args()
    if min(args.base_steps,args.steps)<1:p.error('positive steps required')
    path=ROOT/args.output
    if path.exists():raise FileExistsError(path)
    report={'config':vars(args),'loss':'soft KL(teacher||student), temperature1 over cosine/.1 logits',
            'provenance':capture(['evaluation/consolidation_soft_control.py','evaluation/memory_consolidation.py',
                                  'evaluation/text_facts.py','evaluation/text_checkpoint.py','models/model.py','models/model_v2.py']), 'runs':[]}
    for seed in args.seeds:
        result=run(seed,args);report['runs'].append(result)
        print(json.dumps({'seed':seed,'before':{k:v['accuracy'] for k,v in result['before'].items()},
                          'teacher_top1':{k:v['top1_truth'] for k,v in result['teacher'].items()},
                          'after':{k:{m:d['accuracy'] for m,d in v['evaluation'].items()} for k,v in result['arms'].items()}}),flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
