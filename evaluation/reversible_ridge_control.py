"""Correct explicit retained constraints using frozen learned E; no training.

Uses true labels through caller API, not M teacher. Measures correction management,
not autonomous learned consolidation. Same features/targets for append and replace.
"""
import copy
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from evaluation.text_checkpoint import load
from evaluation.text_facts import frozen_hash
from evaluation.memory_consolidation import questions
from models.feature_readout import FrozenFeatureReadout
from models.cumulative_ridge import CumulativeRidge
from models.reversible_ridge import ReversibleRidge
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(seed,reference):
    base,task,artifact=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt','cpu');base.use_mem=False
    if artifact['sha256']!=reference['checkpoint']['sha256']:raise AssertionError('checkpoint differs')
    original=frozen_hash(base);features=FrozenFeatureReadout(base,'e_key')
    entities=sum(reference['groups'],[]);old_labels=sum(reference['labels'],[])
    code=base.code.double();x_train={e:features.features(questions(task,[e],(2,3))).double() for e in entities}
    x_held={e:features.features(questions(task,[e],(4,5))).double() for e in entities}
    append=CumulativeRidge(96,48);replace=ReversibleRidge(96,48,capacity=12)
    target_by_entity=dict(zip(entities,old_labels));journal=[]
    for e in entities:
        y=code[target_by_entity[e]][None].expand(2,-1)
        append.add(x_train[e],y);replace.put(f'entity:{e}/color',x_train[e],y)
    append.solve()

    def measure(model,targets):
        result={}
        for group,keys in [('changed',entities[:4]),('unchanged',entities[4:])]:
            for name,inputs in [('train',x_train),('held',x_held)]:
                pred=[];truth=[]
                for e in keys:
                    p=(F.normalize(model.predict(inputs[e]),dim=-1)@code.T).argmax(-1)
                    pred.extend(p.tolist());truth.extend([targets[e]]*2)
                result[f'{group}/{name}']={'accuracy':sum(a==b for a,b in zip(pred,truth))/len(truth),
                                          'predicted':pred,'targets':truth}
        return result
    before={'append':measure(append,target_by_entity),'replace':measure(replace,target_by_entity)}
    revised=dict(target_by_entity)
    for e,new in zip(entities[:4],old_labels[:4][-1:]+old_labels[:4][:-1]):
        revised[e]=new;y=code[new][None].expand(2,-1)
        append.add(x_train[e],y);journal.append(replace.put(f'entity:{e}/color',x_train[e],y))
    append.solve()
    after={'append':measure(append,revised),'replace':measure(replace,revised)}
    oracle=CumulativeRidge(96,48)
    for e in entities:oracle.add(x_train[e],code[revised[e]][None].expand(2,-1))
    oracle.solve()
    delta=float((oracle.weight-replace.model.weight).abs().max())
    if delta>1e-7:raise AssertionError('replaced solution differs from latest-only oracle')
    replacement_before_remove=replace.model.weight.clone()
    remove_receipts=[]
    for e in entities[:4]:remove_receipts.append(replace.remove(f'entity:{e}/color'))
    remaining=CumulativeRidge(96,48)
    for e in entities[4:]:remaining.add(x_train[e],code[revised[e]][None].expand(2,-1))
    remaining.solve();remove_delta=float((remaining.weight-replace.model.weight).abs().max())
    if remove_delta>1e-7:raise AssertionError('removal mismatch')
    if frozen_hash(base)!=original or any(m.count_nonzero() for m in base.mem):raise AssertionError('base or M changed')
    return {'seed':seed,'checkpoint':artifact,'before':before,'after_correction':after,
            'replacement_vs_latest_only_max_weight_delta':delta,'remove_vs_remaining_max_weight_delta':remove_delta,
            'replacement_receipts':journal,'remove_receipts':remove_receipts,
            'append_storage':append.storage(),'replacement_storage_after_remove':replace.storage(),
            'frozen_base_unchanged':True,'M_off_zero':True,
            'scope':'direct known-label correction, equal one observation per train surface per update; no stage-SGD comparison'}


def main():
    out=ROOT/'experiments/consolidation/reversible_ridge_correction.json'
    if out.exists():raise FileExistsError(out)
    source=ROOT/'experiments/consolidation/sequence_random_e200.json'
    reference=json.loads(source.read_text(encoding='utf-8'))
    report={'runs':[run(r['seed'],r) for r in reference['runs']],
            'provenance':capture(['models/reversible_ridge.py','models/cumulative_ridge.py','models/feature_readout.py',
                                  'evaluation/reversible_ridge_control.py']),
            'warning':'caller semantic slots and labels, external raw constraints stored, not automatic truth checking or original A consolidation'}
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{ 'seed':r['seed'],'after':{arm:{k:v['accuracy'] for k,v in ev.items()} for arm,ev in r['after_correction'].items()}}
                      for r in report['runs']]))


if __name__=='__main__':main()
