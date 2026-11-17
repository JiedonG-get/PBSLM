"""Train key-only projection on12entities/templates0..3; held4/5 and4names evaluation.

Fixed600updates, no color targets, no test selection. Original E encoder and trained
fast-state params frozen. Compare same initial random projection vs fitted projector.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from models.text_key_projector import TextKeyProjector,key_alignment_loss
from models.feature_readout import FrozenFeatureReadout
from evaluation.text_key_transfer import projection_matrix,counts
from evaluation.text_checkpoint import load
from evaluation.text_facts import frozen_hash
from evaluation.continuous_key_control import state_hash
from runtime.fast_state_online import load_fast_model
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate(table,fast,seed):
    gen=torch.Generator().manual_seed(seed+212000);traces=[];metrics={}
    for support in (8,16):
        for episode in range(16):
            records=torch.randperm(16,generator=gen)[:support]
            colors=torch.randint(8,(support,),generator=gen);st=torch.randint(2,(support,),generator=gen)
            qt=torch.randint(2,(16,),generator=gen)+4
            state=fast.initial(1)
            for e,t,c in zip(records,st,colors):state=fast.observe(state,table[e,t][None],c[None])
            truth=torch.full((16,),8,dtype=torch.long);truth[records]=colors
            q=table[torch.arange(16),qt];score=q@table[records,st].T;confidence,index=score.max(-1)
            prediction={'fast':fast.query(state.expand(16,-1,-1),q).argmax(-1),
                        'nearest_always':colors[index],
                        'nearest_cos.9':torch.where(confidence>=.9,colors[index],torch.full_like(index,8))}
            for name,p in prediction.items():
                for group,indices in [('familiar',torch.arange(12)),('new_names',torch.arange(12,16)),('all',torch.arange(16))]:
                    key=f'{support}/{name}/{group}';c=counts(p[indices],truth[indices])
                    agg=metrics.setdefault(key,{n:0 for n in c})
                    for n,v in c.items():agg[n]+=v
            traces.append({'support':support,'episode':episode,'records':records.tolist(),'support_templates':st.tolist(),
                'query_templates':qt.tolist(),'written_values':colors.tolist(),'targets':truth.tolist(),
                'predictions':{n:p.tolist() for n,p in prediction.items()},'nearest_entity':records[index].tolist(),
                'nearest_cosine':confidence.tolist()})
    return {'metrics':metrics,'traces':traces}


def run(seed,args):
    base,task,text_artifact=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt','cpu');base.use_mem=False
    before=frozen_hash(base);extractor=FrozenFeatureReadout(base)
    fast,fast_artifact=load_fast_model(ROOT/f'experiments/recurrent_state/content_fast600/delta_seed{seed}.pt')
    fast_before=state_hash(fast)
    with torch.no_grad():
        x=torch.stack([task.inputs[e,t] for e in range(16) for t in range(6)])
        raw=F.normalize(extractor.features(x),dim=-1).reshape(16,6,96).detach()
    init=projection_matrix(seed);projector=TextKeyProjector(init)
    opt=torch.optim.AdamW(projector.parameters(),lr=.002);gen=torch.Generator().manual_seed(seed+221000);curve=[]
    for step in range(args.updates):
        st=torch.randint(2,(12,),generator=gen);qt=torch.randint(2,(12,),generator=gen)+2
        support=projector(raw[torch.arange(12),st]);query=projector(raw[torch.arange(12),qt])
        loss,parts=key_alignment_loss(support,query)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(projector.parameters(),1.,error_if_nonfinite=True);opt.step()
        if (step+1)%100==0:curve.append({'step':step+1,'loss':float(loss.detach()),**{n:float(v) for n,v in parts.items()}})
    projector.eval()
    with torch.no_grad():tables={'random':F.normalize(raw@init,dim=-1),'aligned':projector(raw)}
    evaluations={arm:evaluate(table,fast,seed) for arm,table in tables.items()}
    if before!=frozen_hash(base) or fast_before!=state_hash(fast):raise AssertionError('frozen model changed')
    path=ROOT/args.output_dir/f'projection_seed{seed}.pt'
    torch.save({'format':'pbslm.text_key_projector.v1','seed':seed,'updates':args.updates,'weight':projector.weight.detach(),
                'trained_entities':list(range(12)),'trained_templates':[0,1,2,3]},path)
    return {'seed':seed,'text_checkpoint':text_artifact,'fast_checkpoint':fast_artifact,'curve':curve,
            'evaluations':evaluations,'projector_params':1536,'frozen_models_unchanged':True,
            'projector_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope':'held template eval on familiar12/new4 names, supervised identity alignment only'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=600)
    p.add_argument('--output-dir',default='experiments/recurrent_state/text_align600');args=p.parse_args()
    out=ROOT/args.output_dir
    if out.exists():raise FileExistsError(out)
    if args.updates<1:p.error('positive updates required')
    out.mkdir(parents=True)
    report={'config':vars(args),'loss':'bidirectional identityCE cosine/.1 + .1 mean squared entity Gram offidentity',
            'provenance':capture(['models/text_key_projector.py','evaluation/train_text_key_alignment.py',
                                  'models/content_fast_state.py','models/feature_readout.py','evaluation/text_key_transfer.py']), 'runs':[]}
    for seed in (0,1,2):
        r=run(seed,args);report['runs'].append(r)
        print(json.dumps({'seed':seed,'support8':{arm:{k:v for k,v in ev['metrics'].items() if k in ['8/fast/all','8/nearest_always/all']}
                       for arm,ev in r['evaluations'].items()}}),flush=True)
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
