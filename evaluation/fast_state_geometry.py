"""Frozen fast-state key-geometry and repeated-hot-write diagnostic. No training.

Same random draws across rules/geometries. Orthogonal basis, random sphere and
correlated sqrt(.9)*shared+sqrt(.1)*orthogonal keys. Queries never receive truth.
Unknown probes always independently sampled sphere: not geometry-matched negatives.
"""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from models.content_fast_state import ContentFastState
from evaluation.continuous_key_control import state_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]
STEPS=(0,1,8,32,128,512,2048)


def geometry_keys(raw,kind):
    if kind=='random':return F.normalize(raw[:,:8],dim=-1)
    q,_=torch.linalg.qr(raw.transpose(-1,-2))
    basis=q.transpose(-1,-2)
    if kind=='orthogonal':return basis[:,:8]
    if kind=='correlated':return (.9**.5)*basis[:,8:9]+(.1**.5)*basis[:,:8]
    raise ValueError('unknown geometry')


def confusion(pred,target):
    return {'n':target.numel(),'correct':int((pred==target).sum()),
            'false_unknown':int((pred==8).sum()),
            'wrong_value':int(((pred!=target)&(pred!=8)).sum())}


@torch.no_grad()
def run(seed,rule,kind):
    path=ROOT/f'experiments/recurrent_state/content_fast600/{rule}_seed{seed}.pt'
    artifact=torch.load(path,map_location='cpu',weights_only=True)
    if artifact['format']!='pbslm.content_fast_state.research.v1' or artifact['rule']!=rule or artifact['seed']!=seed or artifact['config']!={'key_dim':16,'values':8}:
        raise ValueError('checkpoint contract mismatch')
    if any(not torch.isfinite(x).all() for x in artifact['state_dict'].values()):raise ValueError('nonfinite artifact')
    model=ContentFastState(rule=rule).eval();model.load_state_dict(artifact['state_dict'],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    before=state_hash(model);gen=torch.Generator().manual_seed(seed+171000);batch=32
    keys=geometry_keys(torch.randn(batch,9,16,generator=gen),kind)
    targets=torch.randint(8,(batch,8),generator=gen);state=model.initial(batch)
    unknown=F.normalize(torch.randn(batch,16,generator=gen),dim=-1)
    for e in range(8):state=model.observe(state,keys[:,e],targets[:,e])
    records=[]
    def probe(step):
        saved=state.clone()
        pred=torch.stack([model.query(state,keys[:,e]).argmax(-1) for e in range(8)],1)
        absent=model.query(state,unknown).argmax(-1)
        if not torch.equal(saved,state) or not torch.isfinite(state).all():raise AssertionError('query mutation/nonfinite state')
        records.append({'step':step,'hot':confusion(pred[:,:1],targets[:,:1]),
              'cold':confusion(pred[:,1:],targets[:,1:]),
              'unknown':{'n':batch,'correct_reject':int((absent==8).sum()),'false_accept':int((absent!=8).sum())},
              'predicted':pred.tolist(),'targets':targets.tolist(),'absent_predictions':absent.tolist(),
              'state_norm_mean':float(state.flatten(1).norm(dim=-1).mean())})
    probe(0)
    for step in range(1,max(STEPS)+1):
        values=torch.randint(8,(batch,),generator=gen);targets[:,0]=values
        state=model.observe(state,keys[:,0],values)
        if step in STEPS:probe(step)
    if state_hash(model)!=before:raise AssertionError('weights changed')
    gram=keys@keys.transpose(1,2);mask=~torch.eye(8,dtype=torch.bool)
    return {'seed':seed,'rule':rule,'geometry':kind,'keys':keys.tolist(),
            'offdiag_mean_cosine':float(gram[:,mask].mean()),'records':records,
            'weights_sha256':before,'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'weights_unchanged':True,'history_access':False,'hot_identity':0,'priming_order':list(range(8))}


def main():
    out=ROOT/'experiments/recurrent_state/content_fast600/geometry_hot2048.json'
    if out.exists():raise FileExistsError(out)
    report={'scope':'frozen-state stress, fixed hot0 and priming, geometry shifts beyond training distribution',
            'runs':[],'provenance':capture(['models/content_fast_state.py','evaluation/fast_state_geometry.py'])}
    for seed in (0,1,2):
        for rule in ('delta','additive'):
            for kind in ('orthogonal','random','correlated'):
                r=run(seed,rule,kind);report['runs'].append(r)
                print(json.dumps({'seed':seed,'rule':rule,'geometry':kind,'final':{k:r['records'][-1][k] for k in ('hot','cold','unknown')}}),flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
