"""Calibrate span-novelty guard WITHOUT value labels, then evaluate held text keys.

Calibration32episodes support8 among familiar12, queriestemplates2/3. Testheld4/5 on
16names including4not projector-optimized; supports8/16. Gate fixed at calibration
known95th percentile, no test tuning. Not identity verification, not zero extra state.
"""
import json
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from models.key_span_state import KeySpanState
from models.text_key_projector import TextKeyProjector
from models.feature_readout import FrozenFeatureReadout
from runtime.fast_state_online import load_fast_model
from evaluation.text_checkpoint import load
from evaluation.continuous_key_control import state_hash
from evaluation.text_facts import frozen_hash
from evaluation.text_key_transfer import counts
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


def span_for(keys):
    span=KeySpanState()
    for key in keys:span.observe(key)
    return span


@torch.no_grad()
def run(seed):
    base,task,text_artifact=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt','cpu');base.use_mem=False
    before=frozen_hash(base);extractor=FrozenFeatureReadout(base)
    fast,fast_artifact=load_fast_model(ROOT/f'experiments/recurrent_state/content_fast600/delta_seed{seed}.pt')
    fast_before=state_hash(fast)
    path=ROOT/f'experiments/recurrent_state/text_align600/projection_seed{seed}.pt'
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['format']!='pbslm.text_key_projector.v1' or ck['seed']!=seed:raise ValueError('projection mismatch')
    projector=TextKeyProjector(ck['weight']).eval()
    if not torch.isfinite(projector.weight).all():raise ValueError('nonfinite projector')
    x=torch.stack([task.inputs[e,t] for e in range(16) for t in range(6)])
    table=projector(F.normalize(extractor.features(x),dim=-1).reshape(16,6,96))
    gen=torch.Generator().manual_seed(seed+231000);calibration=[]
    for episode in range(32):
        e=torch.randperm(12,generator=gen)[:8];st=torch.randint(2,(8,),generator=gen);qt=torch.randint(2,(8,),generator=gen)+2
        span=span_for(table[e,st]);scores=span.novelty(table[e,qt])
        calibration.append({'entities':e.tolist(),'support_templates':st.tolist(),'query_templates':qt.tolist(),'scores':scores.tolist()})
    scores=torch.tensor([s for c in calibration for s in c['scores']],dtype=torch.float64)
    threshold=float(torch.quantile(scores,.95,interpolation='higher'))
    gen=torch.Generator().manual_seed(seed+232000);traces=[];metrics={}
    for support in (8,16):
        for episode in range(16):
            e=torch.randperm(16,generator=gen)[:support];st=torch.randint(2,(support,),generator=gen)
            qt=torch.randint(2,(16,),generator=gen)+4;labels=torch.randint(8,(support,),generator=gen)
            span=span_for(table[e,st]);state=fast.initial(1)
            for entity,t,v in zip(e,st,labels):state=fast.observe(state,table[entity,t][None],v[None])
            query=table[torch.arange(16),qt];pred=fast.query(state.expand(16,-1,-1),query).argmax(-1)
            novelty=span.novelty(query);guard=novelty>threshold
            guarded=torch.where(guard,torch.full_like(pred,8),pred)
            target=torch.full((16,),8,dtype=torch.long);target[e]=labels
            for arm,p in [('state',pred),('span_guard',guarded)]:
                for group,indices in [('all',torch.arange(16)),('familiar',torch.arange(12)),('new_names',torch.arange(12,16))]:
                    key=f'{support}/{arm}/{group}';c=counts(p[indices],target[indices]);agg=metrics.setdefault(key,{n:0 for n in c})
                    for n,v in c.items():agg[n]+=v
            traces.append({'support':support,'episode':episode,'entities':e.tolist(),'support_templates':st.tolist(),
                'query_templates':qt.tolist(),'written_values':labels.tolist(),'targets':target.tolist(),
                'state_prediction':pred.tolist(),'guard_prediction':guarded.tolist(),'novelty':novelty.tolist(),
                'gate_reject':guard.tolist(),'span_rank':span.rank})
    if before!=frozen_hash(base) or fast_before!=state_hash(fast):raise AssertionError('frozen model changed')
    return {'seed':seed,'text_checkpoint':text_artifact,'fast_checkpoint':fast_artifact,
        'projection_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'threshold':threshold,'calibration':calibration,'calibration_known_keep':float((scores<=threshold).double().mean()),
        'metrics':metrics,'traces':traces,'frozen_models_unchanged':True,
        'span_state_bytes_per_stream':16*16*8,'fast_state_bytes_per_stream':512}


def main():
    out=ROOT/'experiments/recurrent_state/text_align600/span_guard.json'
    if out.exists():raise FileExistsError(out)
    report={'scope':'extra internal geometry state membership guard, no raw key list/no history retrieval, not identity proof',
        'threshold_rule':'higher95th percentile of known validation residuals only; no test values/unknown labels used',
        'provenance':capture(['models/key_span_state.py','evaluation/text_span_guard.py','models/text_key_projector.py',
                             'models/content_fast_state.py','models/feature_readout.py']), 'runs':[]}
    for seed in (0,1,2):
        r=run(seed);report['runs'].append(r)
        print(json.dumps({'seed':seed,'threshold':r['threshold'],'support8':{k:v for k,v in r['metrics'].items() if k in ('8/state/all','8/span_guard/all')}}),flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
