"""Stage-boundary compression of identical historical ridge constraints.

No neural training; fixed learned E and same three-stage samples/teacher targets.
All ranks process each stage before evaluation; no future features used for basis.
"""
import json
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from models.low_rank_ridge import LowRankRidge
from models.cumulative_ridge import CumulativeRidge
from models.feature_readout import FrozenFeatureReadout
from evaluation.text_checkpoint import load
from evaluation.text_facts import frozen_hash
from evaluation.memory_consolidation import questions
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(ref,steps):
    seed=ref['seed'];base,task,artifact=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt','cpu')
    if artifact['sha256']!=ref['checkpoint']['sha256'] or frozen_hash(base)!=ref['teacher_non_memory_sha256']:
        raise ValueError('checkpoint/reference mismatch')
    before=frozen_hash(base);base.use_mem=False;extractor=FrozenFeatureReadout(base)
    code=base.code.double();labels=[torch.tensor(x) for x in ref['labels']];groups=ref['groups']
    train=[extractor.features(questions(task,g,(2,3))).double().reshape(4,2,-1) for g in groups]
    test={(i,split):extractor.features(questions(task,g,templates)).double()
           for i,g in enumerate(groups) for split,templates in [('train',(2,3)),('held',(4,5))]}
    models={str(rank):LowRankRidge(96,48,rank) for rank in (4,8,16,32,97)}
    joint=CumulativeRidge(96,48);stages=[]
    def measure(model):
        out={}
        for (g,split),features in test.items():
            predicted=(F.normalize(model.predict(features),dim=-1)@code.T).argmax(-1);target=labels[g].repeat_interleave(2)
            out[f'group{g}/{split}']={'accuracy':float((predicted==target).double().mean()),
                                    'predicted':predicted.tolist(),'targets':target.tolist()}
        return out
    for stage in range(3):
        generator=torch.Generator().manual_seed(seed+72000+stage)
        probabilities=torch.tensor(ref['teacher'][stage]['probabilities'],dtype=torch.float64)
        xs=[];ys=[]
        for _ in range(steps):
            i=int(torch.randint(4,(),generator=generator));t=int(torch.randint(2,4,(),generator=generator))-2
            xs.append(train[stage][i,t]);ys.append(code[labels[stage][i]] if stage==0 else probabilities[i,t]@code)
        x=torch.stack(xs);y=torch.stack(ys);joint.add(x,y);joint.solve()
        arms={}
        for name,model in models.items():
            model.add_stage(x,y)
            arms[name]={'evaluation':measure(model),'storage':model.storage(),'compression':model.diagnostics[-1]}
        delta=max(float((models['97'].predict(features)-joint.predict(features)).abs().max()) for features in test.values())
        if delta>1e-6:raise AssertionError('full rank sketch does not match uncompressed ridge')
        stages.append({'stage':stage,'ranks':arms,'full_rank_max_prediction_delta':delta,
                       'uncompressed':measure(joint),'uncompressed_storage':joint.storage()})
    if frozen_hash(base)!=before or any(m.count_nonzero() for m in base.mem):raise AssertionError('base state changed')
    return {'seed':seed,'checkpoint':artifact,'stages':stages,'base_unchanged':True,'M_off_zero':True}


def main():
    source=ROOT/'experiments/consolidation/sequence_random_e200.json';out=ROOT/'experiments/consolidation/rank_compression200.json'
    if out.exists():raise FileExistsError(out)
    ref=json.loads(source.read_text(encoding='utf-8'))
    report={'scope':'stage compression of statistic memory, not original A-only consolidation or language ability',
            'reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'ranks':[4,8,16,32,97],'rank_selection':'fixed ahead of run, not selected for held score',
            'runs':[], 'provenance':capture(['models/low_rank_ridge.py','models/cumulative_ridge.py','models/feature_readout.py',
                                           'evaluation/rank_compression_control.py'])}
    for r in ref['runs']:
        result=run(r,ref['config']['steps']);report['runs'].append(result)
        print(json.dumps({'seed':r['seed'],'final':{rank:{k:v['accuracy'] for k,v in data['evaluation'].items()}
                   for rank,data in result['stages'][-1]['ranks'].items()}}),flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
