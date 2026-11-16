"""Cumulative ridge on exact frozen features/teacher data from three-stage control.

Adds same sampled observations as SGD, solves once/stage. Ridge targets teacher
expected code vectors, different from KL on normalized output: NOT loss-only or
equal-storage/equal-FLOPs ablation. No gradient or original PBSLM parameter update.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from models.feature_readout import FrozenFeatureReadout
from models.cumulative_ridge import CumulativeRidge
from evaluation.text_checkpoint import load
from evaluation.memory_consolidation import questions
from evaluation.text_facts import frozen_hash
from evaluation.sequential_feature_consolidation import randomize_e_only
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(seed,args,ref):
    base,task,artifact=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt','cpu')
    if artifact['sha256']!=ref['checkpoint']['sha256'] or frozen_hash(base)!=ref['teacher_non_memory_sha256']:
        raise ValueError('reference teacher/checkpoint mismatch')
    base.use_mem=False;groups=ref['groups'];labels=[torch.tensor(x) for x in ref['labels']]
    code=base.code.detach().cpu().double();arms={}
    for feature in ('trained_e','random_e','token_mean'):
        candidate=copy.deepcopy(base)
        if feature=='random_e':randomize_e_only(candidate,task,seed+73000)
        torch.manual_seed(seed+74000)
        extractor=FrozenFeatureReadout(candidate,'token_mean' if feature=='token_mean' else 'e_key')
        before=frozen_hash(candidate)
        # Same frozen features across stages; cache avoids repeated neural calls.
        train=[extractor.features(questions(task,g,(2,3))).double().reshape(4,2,-1) for g in groups]
        tests={(i,name):extractor.features(questions(task,g,templates)).double()
               for i,g in enumerate(groups) for name,templates in [('train',(2,3)),('held',(4,5))]}
        ridge=CumulativeRidge(base.e_n*base.d,base.d,regularization=.01)
        stages=[];start=time.perf_counter()
        observations=[]
        for stage in range(3):
            rng=torch.Generator().manual_seed(seed+72000+stage)
            probabilities=torch.tensor(ref['teacher'][stage]['probabilities'],dtype=torch.float64)
            for step in range(args.steps):
                i=int(torch.randint(4,(),generator=rng));t=int(torch.randint(2,4,(),generator=rng))-2
                target=code[labels[stage][i]] if stage==0 else probabilities[i,t]@code
                x=train[stage][i,t][None];y=target[None]
                ridge.add(x,y);observations.append((x.clone(),y.clone()))
            ridge.solve();metrics={}
            for (g,name),features in tests.items():
                scores=F.normalize(ridge.predict(features),dim=-1)@code.T
                prediction=scores.argmax(-1);expected=labels[g].repeat_interleave(2)
                metrics[f'group{g}/{name}']={'accuracy':float((prediction==expected).double().mean()),
                                         'predicted':prediction.tolist(),'targets':expected.tolist()}
            stages.append({'stage':stage,'evaluation':metrics,'storage':ridge.storage()})
        # Reverse-order rebuild of same constraints tests finite-precision order sensitivity.
        reverse=CumulativeRidge(ridge.features,ridge.outputs,regularization=.01)
        for x,y in reversed(observations):reverse.add(x,y)
        reverse.solve()
        delta=float((ridge.weight-reverse.weight).abs().max())
        if delta>1e-7:raise AssertionError('unexpected large numeric order sensitivity')
        if frozen_hash(candidate)!=before or any(m.count_nonzero() for m in candidate.mem):raise AssertionError('base/M altered')
        arms[feature]={'stages':stages,'statistics_fit_seconds_excluding_feature_cache':time.perf_counter()-start,
                       'reverse_order_max_abs_weight_delta':delta,'base_unchanged':True,'M_disabled_zero':True,
                       'gradient_reference_final':ref['arms'][feature]['stages'][-1]['evaluation'],
                       'experimental_raw_constraint_cache_for_order_test_only':len(observations)}
    return {'seed':seed,'checkpoint':artifact,'arms':arms}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--reference',default='experiments/consolidation/sequence_random_e200.json')
    p.add_argument('--steps',type=int,default=200);p.add_argument('--output',default='experiments/consolidation/cumulative_ridge200.json')
    args=p.parse_args();out=ROOT/args.output;source=ROOT/args.reference
    if out.exists():raise FileExistsError(out)
    ref=json.loads(source.read_text(encoding='utf-8'))
    if args.steps!=ref['config']['steps']:raise ValueError('sample budgets differ')
    report={'config':vars(args),'reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'objective':'sum squared error to true/teacher-expected code vectors + .01 times all coefficients squared, including bias',
            'provenance':capture(['models/cumulative_ridge.py','models/feature_readout.py','evaluation/cumulative_ridge_control.py',
                                  'evaluation/sequential_feature_consolidation.py','models/model.py','models/model_v2.py']), 'runs':[]}
    for seed in args.seeds:
        result=run(seed,args,next(x for x in ref['runs'] if x['seed']==seed));report['runs'].append(result)
        print(json.dumps({'seed':seed,'final':{arm:{k:v['accuracy'] for k,v in d['stages'][-1]['evaluation'].items()}
                                                 for arm,d in result['arms'].items()}}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
