"""Disentangle solver versus historical constraints with identical SSE targets.

Learned E features only, frozen model CPU. Three same-loss arms: cumulative solve,
cumulative iterative CG, current-stage-only solve. CG retains all history stats;
not SGD or fair compute benchmark. Reference JSON supplies exact sampled schedule.
"""
import json
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from evaluation.text_checkpoint import load
from evaluation.text_facts import frozen_hash
from evaluation.memory_consolidation import questions
from models.feature_readout import FrozenFeatureReadout
from models.cumulative_ridge import CumulativeRidge
from models.ridge_iterative import solve_cg
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(reference,steps):
    seed=reference['seed'];base,task,artifact=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt','cpu')
    if artifact['sha256']!=reference['checkpoint']['sha256'] or frozen_hash(base)!=reference['teacher_non_memory_sha256']:
        raise ValueError('reference model mismatch')
    before=frozen_hash(base);base.use_mem=False;extractor=FrozenFeatureReadout(base)
    groups=reference['groups'];labels=[torch.tensor(y) for y in reference['labels']];code=base.code.double()
    features=[extractor.features(questions(task,g,(2,3))).double().reshape(4,2,-1) for g in groups]
    tests={(g,split):extractor.features(questions(task,entities,templates)).double()
            for g,entities in enumerate(groups) for split,templates in [('train',(2,3)),('held',(4,5))]}
    cumulative=CumulativeRidge(96,48);warm=None;stages=[]
    def evaluate(model):
        result={}
        for (g,name),x in tests.items():
            predicted=(F.normalize(model.predict(x),dim=-1)@code.T).argmax(-1)
            targets=labels[g].repeat_interleave(2)
            result[f'group{g}/{name}']={'accuracy':float((predicted==targets).double().mean()),
                                      'predicted':predicted.tolist(),'targets':targets.tolist()}
        return result
    for stage in range(3):
        current=CumulativeRidge(96,48);rng=torch.Generator().manual_seed(seed+72000+stage)
        probabilities=torch.tensor(reference['teacher'][stage]['probabilities'],dtype=torch.float64)
        for _ in range(steps):
            i=int(torch.randint(4,(),generator=rng));t=int(torch.randint(2,4,(),generator=rng))-2
            x=features[stage][i,t][None]
            y=(code[labels[stage][i]] if stage==0 else probabilities[i,t]@code)[None]
            cumulative.add(x,y);current.add(x,y)
        cumulative.solve();current.solve()
        warm,convergence=solve_cg(cumulative.gram,cumulative.cross,initial=warm)
        iterative=CumulativeRidge(96,48);iterative.weight=warm.clone()
        delta=float((warm-cumulative.weight).abs().max())
        if not convergence['converged'] or delta>1e-5:raise AssertionError('iterative control failed to match direct optimum')
        def objective(w):
            return float((w*(cumulative.gram@w)).sum()-2*(w*cumulative.cross).sum()+.01*(w*w).sum())
        stages.append({'stage':stage,'cumulative_direct':evaluate(cumulative),'cumulative_CG':evaluate(iterative),
                       'current_only_direct':evaluate(current),'CG':convergence,'max_weight_delta':delta,
                       'cumulative_objective_without_constant':{name:objective(w) for name,w in
                            [('direct',cumulative.weight),('CG',warm),('current_only',current.weight)]},
                       'cumulative_constraints':cumulative.samples,'current_constraints':current.samples})
    if frozen_hash(base)!=before or any(m.count_nonzero() for m in base.mem):raise AssertionError('base changed')
    return {'seed':seed,'stages':stages,'base_unchanged':True,'M_off_zero':True}


def main():
    source=ROOT/'experiments/consolidation/sequence_random_e200.json'
    out=ROOT/'experiments/consolidation/same_loss_history200.json'
    if out.exists():raise FileExistsError(out)
    data=json.loads(source.read_text(encoding='utf-8'))
    report={'reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':'same regularized SSE; direct vs full-history CG isolates solver, current-only changes retained constraints',
            'runs':[run(r,data['config']['steps']) for r in data['runs']],
            'provenance':capture(['models/ridge_iterative.py','models/cumulative_ridge.py','models/feature_readout.py',
                                 'evaluation/ridge_same_loss_control.py'])}
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'seed':r['seed'],'final':{name:{k:v['accuracy'] for k,v in r['stages'][-1][name].items()}
                       for name in ('cumulative_direct','cumulative_CG','current_only_direct')},
                       'max_CG_delta':max(s['max_weight_delta'] for s in r['stages'])} for r in report['runs']]))


if __name__=='__main__':main()
