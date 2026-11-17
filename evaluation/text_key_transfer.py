"""Frozen text E -> fixed16D projection -> saved fast-state transfer diagnostic.

No projection/encoder/readout training; random projection seed fixed in code. Tests
representation/decoder distribution compatibility, not language understanding.
Explicit observed color classes supplied at write. Source text model independently
trained for small template task; fast model trained on random sphere keys.
"""
import json
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from evaluation.text_checkpoint import load
from evaluation.continuous_key_control import state_hash
from evaluation.text_facts import frozen_hash,ENTITIES,TEMPLATES
from models.feature_readout import FrozenFeatureReadout
from runtime.fast_state_online import load_fast_model
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


def projection_matrix(seed,source_dim=96,target_dim=16):
    gen=torch.Generator().manual_seed(seed+211000)
    q,_=torch.linalg.qr(torch.randn(source_dim,target_dim,generator=gen,dtype=torch.float64))
    return q.float()


def counts(pred,target):
    known=target!=8;unknown=~known
    return {'known_n':int(known.sum()),'known_correct':int(((pred==target)&known).sum()),
            'known_false_unknown':int(((pred==8)&known).sum()),
            'known_wrong_value':int(((pred!=target)&(pred!=8)&known).sum()),
            'unknown_n':int(unknown.sum()),'unknown_false_accept':int(((pred!=8)&unknown).sum()),
            'unknown_correct_reject':int(((pred==8)&unknown).sum())}


@torch.no_grad()
def run(seed):
    base,task,text_artifact=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt','cpu')
    base.use_mem=False;before=frozen_hash(base);extractor=FrozenFeatureReadout(base)
    fast,fast_artifact=load_fast_model(ROOT/f'experiments/recurrent_state/content_fast600/delta_seed{seed}.pt')
    fast_before=state_hash(fast)
    inputs=torch.stack([task.inputs[e,t] for e in range(16) for t in range(6)])
    raw=F.normalize(extractor.features(inputs),dim=-1).reshape(16,6,96)
    project=projection_matrix(seed)
    keys=F.normalize(raw@project,dim=-1)
    gen=torch.Generator().manual_seed(seed+212000);traces=[];metrics={}
    for support in (4,8,12,16):
        for episode in range(16):
            order=torch.randperm(16,generator=gen);records=order[:support]
            colors=torch.randint(8,(support,),generator=gen)
            st=torch.randint(2,(support,),generator=gen)
            state=fast.initial(1)
            for entity,template,color in zip(records,st,colors):
                state=fast.observe(state,keys[entity,template][None],color[None])
            truth=torch.full((16,),8,dtype=torch.long);truth[records]=colors
            for split in ('same_surface','held'):
                qt=torch.randint(2,(16,),generator=gen)+4 if split=='held' else torch.zeros(16,dtype=torch.long)
                if split=='same_surface':qt[records]=st
                query_keys=keys[torch.arange(16),qt]
                fast_pred=fast.query(state.expand(16,-1,-1),query_keys).argmax(-1)
                predictions={'fast16':fast_pred}
                geometry={}
                for name,table in [('nearest96',raw),('nearest16',keys)]:
                    record_keys=table[records,st];queries=table[torch.arange(16),qt]
                    sim=queries@record_keys.T;confidence,top=sim.max(-1)
                    predictions[name+'_always']=colors[top]
                    predictions[name+'_cos.9']=torch.where(confidence>=.9,colors[top],torch.full_like(top,8))
                    geometry[name]={'top_record_entity':records[top].tolist(),'max_cosine':confidence.tolist()}
                row={'support':support,'episode':episode,'split':split,'record_entities':records.tolist(),
                     'record_templates':st.tolist(),'record_values':colors.tolist(),'query_templates':qt.tolist(),
                     'targets':truth.tolist(),'geometry':geometry,'predictions':{k:v.tolist() for k,v in predictions.items()}}
                traces.append(row)
                for method,pred in predictions.items():
                    key=f'{support}/{split}/{method}';c=counts(pred,truth)
                    aggregate=metrics.setdefault(key,{name:0 for name in c})
                    for name,value in c.items():aggregate[name]+=value
    if before!=frozen_hash(base) or fast_before!=state_hash(fast):raise AssertionError('model changed')
    return {'seed':seed,'text_checkpoint':text_artifact,'fast_checkpoint':fast_artifact,
            'projection_seed':seed+211000,'projection_shape':[96,16],
            'projection_sha256':hashlib.sha256(project.numpy().tobytes()).hexdigest(),
            'metrics':metrics,'traces':traces,'models_unchanged':True,
            'identity_interface':'entity indices used only for input lookup and scoring, not in fast-state observe/query',
            'limits':'unwritten entity queries use same16name inventory;4 names untrained by source episodic optimization'}


def main():
    out=ROOT/'experiments/recurrent_state/content_fast600/text_key_transfer.json'
    if out.exists():raise FileExistsError(out)
    report={'scope':'untrained cross-model key projection, explicit value feedback, no semantic extraction or language generation',
            'design':'fixed seed singleprojection per model,16episodes/support4,8,12,16;no rank/threshold search',
            'provenance':capture(['evaluation/text_key_transfer.py','models/feature_readout.py','models/content_fast_state.py',
                                 'evaluation/text_facts.py','runtime/fast_state_online.py']), 'runs':[]}
    for seed in (0,1,2):
        r=run(seed);report['runs'].append(r)
        print(json.dumps({'seed':seed,'support8held':{k:v for k,v in r['metrics'].items() if k.startswith('8/held/')}}),flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
