"""Paired fixed-key representation robustness; remove memory-budget confounding.

Same16 entity labels across original/repeated/prefix/suffix layouts. Repetition
changes length AND multiplicity, not pure length-only causality. Prefix/suffix
contains another entity question treated by experimental policy as irrelevant.
No instruction/role marker teaches that policy to model; not NLP reasoning.
"""
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from models.feature_readout import FrozenFeatureReadout
from models.cumulative_ridge import CumulativeRidge
from models.reservoir_nearest import ReservoirNearest
from evaluation.text_checkpoint import load
from evaluation.text_facts import frozen_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]
LAYOUTS=('original','repeat','prefix','suffix')


def layout_tokens(task,entity,template,layout):
    target=task.inputs[entity,template]
    # Different distractor for every target, fixed across train/held surfaces.
    distractor=task.inputs[(entity+1)%16,template]
    if layout=='original':return target
    if layout=='repeat':return torch.cat((target,target))
    if layout=='prefix':return torch.cat((distractor,target))
    if layout=='suffix':return torch.cat((target,distractor))
    raise ValueError('unknown layout')


@torch.no_grad()
def run(seed):
    base,task,artifact=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt','cpu')
    before=frozen_hash(base);base.use_mem=False;extractor=FrozenFeatureReadout(base)
    gen=torch.Generator().manual_seed(seed+91000)
    label_per_entity=task.labels[torch.randperm(16,generator=gen)%8]
    labels=label_per_entity.repeat_interleave(2);code=base.code.double();arms={};origin=None
    for layout in LAYOUTS:
        def encode(templates):
            x=torch.stack([layout_tokens(task,e,t,layout) for e in range(16) for t in templates])
            return extractor.features(x).double()
        train=encode((2,3));held=encode((4,5))
        ridge=CumulativeRidge(96,48);ridge.add(train,code[labels]);ridge.solve()
        nearest=ReservoirNearest(96,len(train)*776);nearest.add(train,labels)
        if layout=='original':
            original_ridge=ridge;original_nearest=nearest
        predictions={'ridge':{},'nearest':{},'original_ridge_no_refit':{},'original_nearest_no_refit':{}}
        for split,features in [('train',train),('held',held)]:
            ridge_pred=(F.normalize(ridge.predict(features),dim=-1)@code.T).argmax(-1)
            nearest_pred=nearest.predict(features)
            original_ridge_pred=(F.normalize(original_ridge.predict(features),dim=-1)@code.T).argmax(-1)
            original_nearest_pred=original_nearest.predict(features)
            for name,predicted in [('ridge',ridge_pred),('nearest',nearest_pred),
                                   ('original_ridge_no_refit',original_ridge_pred),('original_nearest_no_refit',original_nearest_pred)]:
                predictions[name][split]={}
                for group,indices in [('all',torch.arange(32)),('familiar',torch.arange(24)),('untrained_names',torch.arange(24,32))]:
                    predictions[name][split][group]={'accuracy':float((predicted[indices]==labels[indices]).double().mean()),
                        'predicted':predicted[indices].tolist(),'targets':labels[indices].tolist()}
        if origin is None:origin=(train.clone(),held.clone())
        similarity={'train_to_original_mean_cosine':float(F.cosine_similarity(train,origin[0],dim=-1).mean()),
                    'held_to_original_mean_cosine':float(F.cosine_similarity(held,origin[1],dim=-1).mean())}
        arms[layout]={'input_length':task.length if layout=='original' else 2*task.length,
                      'evaluation':predictions,'feature_similarity':similarity,
                      'ridge_storage':ridge.storage(),'nearest_storage':nearest.storage()}
    if before!=frozen_hash(base) or any(m.count_nonzero() for m in base.mem):raise AssertionError('base changed')
    return {'seed':seed,'checkpoint':artifact,'label_per_entity':label_per_entity.tolist(),'arms':arms,'base_unchanged':True,'M_off_zero':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='experiments/consolidation/layout_robustness16_transfer.json')
    args=parser.parse_args()
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    report={'scope':'paired frozen-feature layout robustness, unbudgeted readouts refit per layout, NOT language comprehension',
             'design':'same entity labels/mapping across layouts, no capacity limit, no teacher/neural training',
             'provenance':capture(['evaluation/layout_robustness_control.py','models/feature_readout.py','models/reservoir_nearest.py',
                                   'models/cumulative_ridge.py','models/model.py','models/model_v2.py']), 'runs':[]}
    for seed in (0,1,2):
        result=run(seed);report['runs'].append(result)
        print(json.dumps({'seed':seed,'held':{k:{method:v['evaluation'][method]['held']['all']['accuracy'] for method in ('ridge','nearest','original_ridge_no_refit','original_nearest_no_refit')}
                                         for k,v in result['arms'].items()}}),flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
