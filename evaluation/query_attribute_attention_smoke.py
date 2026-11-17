"""Simple query/attribute attention mechanism check, no learned encoder.

Hand-designed factorized keys make object and attribute explicitly visible. Reuses
standard cosine attention. Positive result is an implementation sanity check, NOT
learned generalization, new architecture superiority, or tokenizer-free cognition.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from models.segmented_working_read import read_segments


def trial(seed,objects=8,attributes=3,classes=8):
    gen=torch.Generator().manual_seed(seed)
    entity=torch.arange(objects).repeat_interleave(attributes)
    attribute=torch.arange(attributes).repeat(objects)
    # Unique values within each object avoid accidental success of attribute-blind reading.
    labels=torch.cat([torch.randperm(classes,generator=gen)[:attributes] for _ in range(objects)])
    keys=torch.cat((F.one_hot(entity,objects),F.one_hot(attribute,attributes)),dim=-1).float()
    values=F.one_hot(labels,classes).float()
    def read(q,k=keys,v=values):
        result=read_segments(q,k[None].expand(len(q),-1,-1),v[None].expand(len(q),-1,-1),temperature=.1)
        return result['value'].argmax(-1),result['weights']
    predicted,weights=read(keys)
    blind=keys.clone();blind[:,objects:]=0
    blind_pred,_=read(blind)
    # Remove entity information: attribute alone cannot identify the requested object.
    attr_only=keys.clone();attr_only[:,:objects]=0
    attr_pred,_=read(attr_only)
    order=torch.randperm(len(keys),generator=gen)
    permuted,permuted_weights=read(keys,keys[order],values[order])
    if not torch.allclose(weights,permuted_weights[:,torch.argsort(order)],atol=1e-6):
        raise AssertionError('attention changed under record permutation')
    # Correct one object/attribute, explicitly retaining all other value records.
    revised=values.clone();new=(int(labels[0])+1)%classes;revised[0]=F.one_hot(torch.tensor(new),classes).float()
    revised_pred,_=read(keys,v=revised)
    expected=labels.clone();expected[0]=new
    # Duplicate unrelated other-object records: target object0 queries stay correct.
    distractors=torch.arange(attributes,len(keys)).repeat(4)
    expanded_keys=torch.cat((keys,keys[distractors]));expanded_values=torch.cat((values,values[distractors]))
    expanded_pred,_=read(keys[:attributes],expanded_keys,expanded_values)
    # Ordinary attention forgets no invalid records by itself: retain old value plus
    # equally keyed contradictory new record, measure ambiguity rather than assert correctness.
    conflict_keys=torch.cat((keys,keys[:1]));conflict_values=torch.cat((values,revised[:1]))
    conflict=read_segments(keys[:1],conflict_keys[None],conflict_values[None],temperature=.1)
    old_score=float(conflict['value'][0,labels[0]]);new_score=float(conflict['value'][0,new])
    return {'seed':seed,'queries':len(keys),'query_attribute_correct':int((predicted==labels).sum()),
            'object_only_correct':int((blind_pred==labels).sum()),'attribute_only_correct':int((attr_pred==labels).sum()),
            'permuted_correct':int((permuted==labels).sum()),
            'correction_correct':int((revised_pred==expected).sum()),
            'untouched_value_records_bitwise_unchanged':torch.equal(values[1:],revised[1:]),
            'unrelated_duplicates_correct':int((expanded_pred==labels[:attributes]).sum()),
            'unrelated_queries':attributes,'contradictory_equal_key_old_score':old_score,
            'contradictory_equal_key_new_score':new_score,
            'labels':labels.tolist(),'predicted':predicted.tolist(),'object_only_predicted':blind_pred.tolist(),
            'query0_attention':weights[0].tolist()}


def main():
    path=Path('experiments/query_attribute_attention/smoke.json')
    if path.exists():raise FileExistsError(path)
    runs=[trial(seed) for seed in (0,1,2)]
    for row in runs:
        if row['query_attribute_correct']!=row['queries'] or row['correction_correct']!=row['queries']:
            raise AssertionError('explicit query/attribute mechanism failed')
        if row['unrelated_duplicates_correct']!=row['unrelated_queries']:raise AssertionError('distractor test failed')
    report={'scope':'hand-designed symbolic query/attribute keys, standard attention; no training',
            'objects':8,'attributes':3,'classes':8,'temperature':.1,'runs':runs}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'total_queries':sum(r['queries'] for r in runs),
                     'query_attribute_correct':sum(r['query_attribute_correct'] for r in runs),
                     'object_only_correct':sum(r['object_only_correct'] for r in runs),
                     'attribute_only_correct':sum(r['attribute_only_correct'] for r in runs),
                     'correction_correct':sum(r['correction_correct'] for r in runs),
                     'unrelated_duplicate_correct':sum(r['unrelated_duplicates_correct'] for r in runs)}))


if __name__=='__main__':main()
