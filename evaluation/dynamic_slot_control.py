"""Dynamic explicit-key state reference, no identity IDs or new training.

Reuses exact continuous-key benchmark with known/unknown groups. Stored keys are
explicit evidence, 26float per slot; larger activation state than fast128matrix.
Frozen learned value embedding/MLP reused but allocation is standard algorithmic.
"""
import hashlib
import json
from pathlib import Path
import torch
from models.dynamic_content_slots import DynamicContentSlots
from runtime.fast_state_online import load_fast_model
from evaluation.continuous_key_control import episode,state_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(seed):
    fast,artifact=load_fast_model(ROOT/f'experiments/recurrent_state/content_fast600/delta_seed{seed}.pt')
    before=state_hash(fast);evaluations={}
    for capacity in (4,8):
        model=DynamicContentSlots(fast,capacity,threshold=.9)
        for noise in (0.,.05,.15):
            gen=torch.Generator().manual_seed(seed+152000+96)
            _,metric=episode(model,gen,64,96,'cpu',noise=noise)
            if not metric['state_finite']:raise AssertionError('nonfinite state')
            evaluations[f'capacity{capacity}/noise{noise}']=metric
    if state_hash(fast)!=before:raise AssertionError('reader weights changed')
    return {'seed':seed,'checkpoint':artifact,'weights_sha256':before,'evaluations':evaluations,
            'slot_state_bytes_per_stream':{str(c):c*26*4 for c in (4,8)},
            'fast_matrix_bytes_per_stream_reference':512,'weights_unchanged':True,
            'no_raw_event_history_access':True,'explicit_key_rows_retained':True}


def main():
    out=ROOT/'experiments/recurrent_state/content_fast600/dynamic_slots.json'
    if out.exists():raise FileExistsError(out)
    report={'scope':'algorithmic dynamic slot allocation with frozen neural value decoding; not learned allocation/general language',
            'protocol':'cos.9match -> replace key/value, else free slot or least-recent-write eviction; query cos.9elseUNKNOWN',
            'provenance':capture(['models/dynamic_content_slots.py','evaluation/dynamic_slot_control.py',
                                  'evaluation/continuous_key_control.py','models/content_fast_state.py']), 'runs':[]}
    for seed in (0,1,2):
        result=run(seed);report['runs'].append(result)
        print(json.dumps({'seed':seed,'counts':{k:v['learned'] for k,v in result['evaluations'].items()}}),flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
