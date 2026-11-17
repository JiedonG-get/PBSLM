"""Paired geometry/value-conflict reference using previously saved delta weights.

Dual baseline explicitly knows observed key record indices; it stores keys/values.
No unseen-key handling or language/compute advantage claim. All keys observed in
priming before baseline construction, no future key use. No retraining/parameter update.
"""
import json
import hashlib
from pathlib import Path
import torch
from models.content_fast_state import ContentFastState
from models.dual_key_reference import DualKeyReference
from evaluation.fast_state_geometry import geometry_keys,confusion
from evaluation.continuous_key_control import state_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(seed,geometry,schedule):
    path=ROOT/f'experiments/recurrent_state/content_fast600/delta_seed{seed}.pt'
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if (ck['format'],ck['rule'],ck['config'])!=('pbslm.content_fast_state.research.v1','delta',{'key_dim':16,'values':8}):
        raise ValueError('unexpected delta artifact')
    model=ContentFastState().eval();model.load_state_dict(ck['state_dict'],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    before=state_hash(model);gen=torch.Generator().manual_seed(seed+181000);batch=32
    keys=geometry_keys(torch.randn(batch,9,16,generator=gen),geometry)
    initial_values=torch.randint(8,(batch,8),generator=gen)
    if schedule=='initially_same':initial_values=initial_values[:,:1].expand(-1,8).clone()
    elif schedule!='independent':raise ValueError('unknown value schedule')
    truth=initial_values.clone();state=model.initial(batch)
    for e in range(8):state=model.observe(state,keys[:,e],truth[:,e])
    # Baseline gets only primed observed keys and value embeddings, never future writes.
    ref=DualKeyReference(keys,model.value_embedding(truth))
    records=[]
    def probe(step):
        delta_scores=torch.stack([model.query(state,keys[:,e]) for e in range(8)],1)
        dual_reads=ref.read(keys)
        # Retain original learned decoder, so linear read recovery != automatic class accuracy.
        ref_scores=model.read(dual_reads.float())
        pred=delta_scores.argmax(-1);other=ref_scores.argmax(-1)
        targets=model.value_embedding(truth).double()
        delta_reads=torch.einsum('bkd,bdv->bkv',keys,state).double()
        row={'step':step,'targets':truth.tolist(),'delta_predictions':pred.tolist(),'dual_predictions':other.tolist(),
             'delta_hot':confusion(pred[:,:1],truth[:,:1]),'delta_cold':confusion(pred[:,1:],truth[:,1:]),
             'dual_hot':confusion(other[:,:1],truth[:,:1]),'dual_cold':confusion(other[:,1:],truth[:,1:]),
             'delta_cold_read_rmse':float((delta_reads[:,1:]-targets[:,1:]).square().mean().sqrt()),
             'dual_read_max_error':float((dual_reads-targets).abs().max())}
        if row['dual_read_max_error']>1e-7:raise AssertionError('dual observed-key recovery failed')
        records.append(row)
    probe(0)
    for step in range(1,513):
        values=torch.randint(8,(batch,),generator=gen);truth[:,0]=values
        state=model.observe(state,keys[:,0],values)
        ref.overwrite(0,model.value_embedding(values))
        if step in (1,8,32,128,512):probe(step)
    if state_hash(model)!=before:raise AssertionError('weights changed')
    return {'seed':seed,'geometry':geometry,'value_schedule':schedule,'records':records,
            'condition_number_max':ref.condition,'reference_storage_per_stream':ref.storage_per_stream(),
            'delta_state_bytes_fp32':512,'weights_sha256':before,
            'checkpoint_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'weights_unchanged':True,'reference_record_index_supplied_by_caller':True}


def main():
    out=ROOT/'experiments/recurrent_state/content_fast600/dual_geometry512.json'
    if out.exists():raise FileExistsError(out)
    report={'scope':'observed-key geometric reference, explicit larger storage, same trained delta decoder',
        'value_schedule_note':'initially_same means same labels only at priming; subsequent hot updates generally conflict with cold',
        'provenance':capture(['models/dual_key_reference.py','models/content_fast_state.py','evaluation/fast_state_geometry.py',
                             'evaluation/dual_geometry_control.py']), 'runs':[]}
    for seed in (0,1,2):
        for geometry in ('orthogonal','random','correlated'):
            for schedule in ('independent','initially_same'):
                r=run(seed,geometry,schedule);report['runs'].append(r)
                print(json.dumps({'seed':seed,'geometry':geometry,'schedule':schedule,
                                 'final_delta_cold':r['records'][-1]['delta_cold'],'final_dual_cold':r['records'][-1]['dual_cold']}),flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
