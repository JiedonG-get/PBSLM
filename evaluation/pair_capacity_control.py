"""128 pair-key arbitrary associations: byte-budget capacity diagnostic, NOT NLP.

Each key concatenates two separately padded existing entity questions (2x old input
length), with one random label for the PAIR. Two training/held surfaces per pair.
All16 entity inventory includes4 names unseen in episodic optimization; unordered
pairs give one canonical ordering per key. Reverse-order behavior is not tested.
"""
import json
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F

from evaluation.text_checkpoint import load
from evaluation.text_facts import frozen_hash
from models.feature_readout import FrozenFeatureReadout
from models.cumulative_ridge import CumulativeRidge
from models.low_rank_ridge import LowRankRidge
from models.reservoir_nearest import ReservoirNearest
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]
RANKS=(8,16,32)


@torch.no_grad()
def run(seed):
    base,task,artifact=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt','cpu');before=frozen_hash(base)
    base.use_mem=False;extractor=FrozenFeatureReadout(base)
    gen=torch.Generator().manual_seed(seed+81000)
    pairs=torch.combinations(torch.arange(16),r=2,with_replacement=True)
    pairs=pairs[torch.randperm(len(pairs),generator=gen)[:128]].tolist()
    labels=task.labels[torch.randperm(128,generator=gen)%8].repeat_interleave(2)
    def pair_features(templates):
        inputs=torch.stack([torch.cat((task.inputs[a,t],task.inputs[b,t])) for a,b in pairs for t in templates])
        if len({tuple(row.tolist()) for row in inputs})!=len(inputs):raise AssertionError('pair inputs collided')
        return torch.cat([extractor.features(inputs[i:i+32]).double() for i in range(0,len(inputs),32)])
    train=pair_features((2,3));held=pair_features((4,5));code=base.code.double()
    # Pair labels independent of constituent entity colors. No linguistic semantics claimed.
    sketches={};retrievers={};budgets={}
    for rank in RANKS:
        sketches[rank]=LowRankRidge(96,48,rank)
        budgets[rank]=(97*rank+rank+rank*48)*8
        retrievers[rank]=ReservoirNearest(96,budgets[rank],seed=seed+82000)
    joint=CumulativeRidge(96,48)
    full_nn=ReservoirNearest(96,256*(96*8+8),seed=0)
    traces=[]
    for stage in range(8):
        start=stage*32;end=start+32;x=train[start:end];y=labels[start:end];values=code[y]
        joint.add(x,values);joint.solve();full_nn.add(x,y)
        for rank in RANKS:
            sketches[rank].add_stage(x,values);retrievers[rank].add(x,y)
        evaluations={}
        for name,model,kind in [('unbudgeted_ridge',joint,'ridge'),('unbudgeted_nearest',full_nn,'nearest')]+[
            (f'{kind}_r{rank}',model,kind) for rank in RANKS for kind,model in [('ridge',sketches[rank]),('nearest',retrievers[rank])]]:
            metrics={}
            for split,features in [('train',train[:end]),('held',held[:end])]:
                predicted=(F.normalize(model.predict(features),dim=-1)@code.T).argmax(-1) if kind=='ridge' else model.predict(features)
                metrics[split]={'accuracy':float((predicted==labels[:end]).double().mean()),
                                'predicted':predicted.tolist(),'targets':labels[:end].tolist()}
            metrics['storage']=model.storage()
            evaluations[name]=metrics
        for rank in RANKS:
            if sketches[rank].storage()['tensor_payload_bytes']>budgets[rank] or retrievers[rank].storage()['tensor_payload_bytes']>budgets[rank]:
                raise AssertionError('tensor budget exceeded')
        traces.append({'stage':stage,'seen_pair_keys':end//2,'evaluation':evaluations})
    if before!=frozen_hash(base) or any(m.count_nonzero() for m in base.mem):raise AssertionError('base state changed')
    return {'seed':seed,'checkpoint':artifact,'pair_keys':pairs,'label_ids_per_pair':labels[::2].tolist(),
            'input_tokens':task.length*2,'original_training_length':task.length,
            'budgets':budgets,'stages':traces,'M_off_zero':True,'frozen_base_unchanged':True,
            'warning':'all pair/held features cached in evaluator, per-method tensor budgets exclude research cache'}


def main():
    out=ROOT/'experiments/consolidation/pair_capacity128.json'
    if out.exists():raise FileExistsError(out)
    report={'scope':'random pair-key binding outside training length; memory/feature stress not meaningful Chinese reasoning',
            'fixed_design':'128 pair keys,2train+2held surfaces,8chunks16pairs,no rank/hyperparameter search',
            'provenance':capture(['models/reservoir_nearest.py','models/low_rank_ridge.py','models/cumulative_ridge.py',
                                  'models/feature_readout.py','evaluation/pair_capacity_control.py']), 'runs':[]}
    for seed in (0,1,2):
        result=run(seed);report['runs'].append(result)
        print(json.dumps({'seed':seed,'final':{k:{s:v[s]['accuracy'] for s in ('train','held')}
             for k,v in result['stages'][-1]['evaluation'].items()}}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
