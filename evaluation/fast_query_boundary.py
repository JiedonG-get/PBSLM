"""Query-noise and near-unknown boundary with frozen delta and explicit references.

No training, same observed writes for all methods. 'Noisy known' vs 'near unknown'
identity semantics supplied by evaluator; vectors alone do not guarantee a boundary.
Threshold.9 and noise/cosine settings fixed, no calibration or efficiency claim.
"""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from models.content_fast_state import ContentFastState
from models.dual_key_reference import DualKeyReference
from evaluation.fast_state_geometry import geometry_keys
from evaluation.continuous_key_control import state_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


def nearby_key(key,raw,cosine):
    if not -1<cosine<1:raise ValueError('cosine strictly between -1 and 1 required')
    k=F.normalize(key,dim=-1);orth=raw-(raw*k).sum(-1,keepdim=True)*k
    if (orth.norm(dim=-1)<1e-6).any():raise ValueError('nonparallel perturbation required')
    return cosine*k+(1-cosine*cosine)**.5*F.normalize(orth,dim=-1)


def confusion(predicted,target):
    known=target!=8;unknown=~known
    return {'n':target.numel(),'known_n':int(known.sum()),'known_correct':int(((predicted==target)&known).sum()),
            'known_false_unknown':int(((predicted==8)&known).sum()),
            'known_wrong_value':int(((predicted!=target)&(predicted!=8)&known).sum()),
            'unknown_n':int(unknown.sum()),'unknown_false_accept':int(((predicted!=8)&unknown).sum()),
            'unknown_correct_reject':int(((predicted==8)&unknown).sum())}


@torch.no_grad()
def run(seed,geometry):
    path=ROOT/f'experiments/recurrent_state/content_fast600/delta_seed{seed}.pt'
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if (ck['rule'],ck['config'])!=('delta',{'key_dim':16,'values':8}):raise ValueError('artifact config mismatch')
    model=ContentFastState().eval();model.load_state_dict(ck['state_dict'],strict=True)
    for parameter in model.parameters():parameter.requires_grad_(False)
    before=state_hash(model);gen=torch.Generator().manual_seed(seed+191000);batch=32
    keys=geometry_keys(torch.randn(batch,9,16,generator=gen),geometry)
    labels=torch.randint(8,(batch,8),generator=gen);state=model.initial(batch)
    for entity in range(8):state=model.observe(state,keys[:,entity],labels[:,entity])
    ref=DualKeyReference(keys,model.value_embedding(labels))
    for _ in range(128):
        value=torch.randint(8,(batch,),generator=gen);labels[:,0]=value
        state=model.observe(state,keys[:,0],value);ref.overwrite(0,model.value_embedding(value))
    perturb=torch.randn(batch,8,16,generator=gen)
    fresh=F.normalize(torch.randn(batch,8,16,generator=gen),dim=-1)
    conditions=[('known_noise0',keys,labels),
                ('known_noise.05',F.normalize(keys+.05*perturb,dim=-1),labels),
                ('known_noise.15',F.normalize(keys+.15*perturb,dim=-1),labels),
                ('unknown_random',fresh,torch.full_like(labels,8)),
                ('unknown_near.95',nearby_key(keys,perturb,.95),torch.full_like(labels,8)),
                ('unknown_near.99',nearby_key(keys,perturb,.99),torch.full_like(labels,8))]
    rows=[]
    for name,query,target in conditions:
        saved=state.clone()
        raw_reads=torch.einsum('bkd,bdv->bkv',query,state)
        delta=model.read(raw_reads).argmax(-1)
        dual=model.read(ref.read(query).float()).argmax(-1)
        similarity=query@keys.transpose(1,2);confidence,idx=similarity.max(-1)
        selected=labels.gather(1,idx)
        nearest=torch.where(confidence>=.9,selected,torch.full_like(selected,8))
        if not torch.equal(saved,state):raise AssertionError('query modified state')
        row={'condition':name,'targets':target.tolist(),
             'nearest_max_cosine':confidence.tolist(),'nearest_record':idx.tolist(),
             'query_source_key_cosine':(query*keys).sum(-1).tolist(),'methods':{}}
        for method,predicted in [('delta',delta),('dual',dual),('nearest_cos.9',nearest)]:
            row['methods'][method]={'counts':confusion(predicted,target),'predicted':predicted.tolist()}
        rows.append(row)
    if before!=state_hash(model):raise AssertionError('weights changed')
    return {'seed':seed,'geometry':geometry,'conditions':rows,'keys':keys.tolist(),'observed_labels':labels.tolist(),
            'checkpoint_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'weights_sha256':before,
            'weights_unchanged':True,'write_updates_after_priming':128}


def main():
    out=ROOT/'experiments/recurrent_state/content_fast600/query_boundary.json'
    if out.exists():raise FileExistsError(out)
    report={'scope':'fixed-weight query boundary stress; evaluator labels distinguish noisy identity from nearby new identity',
            'reference_threshold':.9,'provenance':capture(['evaluation/fast_query_boundary.py','models/dual_key_reference.py',
                           'models/content_fast_state.py','evaluation/fast_state_geometry.py']), 'runs':[]}
    for seed in (0,1,2):
        for geometry in ('random','correlated'):report['runs'].append(run(seed,geometry))
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Saved six runs of six query conditions for three fixed read methods.')


if __name__=='__main__':main()
