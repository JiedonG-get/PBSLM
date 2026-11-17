"""Same-weight same-initial-state hot-write gate intervention; no history access."""
import json
from pathlib import Path
import torch
from models.state_write_intervention import observe_with_gate
from evaluation.working_state_checkpoint import load_working_state
from evaluation.working_state_skew import CHECKPOINTS
from evaluation.recurrent_state_control import weights_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(seed,batch=32):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,metadata=load_working_state(ROOT/f'experiments/recurrent_state/slots600/learned_seed{seed}.pt',device)
    before=weights_hash(model);generator=torch.Generator().manual_seed(seed+121000)
    truth=torch.randint(8,(batch,8),generator=generator).to(device);initial=model.initial(batch)
    for entity in range(8):initial=model.observe(initial,torch.full((batch,),entity,device=device,dtype=torch.long),truth[:,entity])
    states={mode:initial.clone() for mode in ('soft','top1_mass','top1_unit')};records={mode:[] for mode in states}
    route=model.routing(torch.arange(8,device=device));top=route.argmax(-1);hot_slot=int(top[0])
    def probe(step):
        for mode,state in states.items():
            saved=state.clone()
            predicted=torch.stack([model.query(state,torch.full((batch,),e,device=device,dtype=torch.long)).argmax(-1) for e in range(8)],1)
            if not torch.equal(state,saved):raise AssertionError('query changed state')
            other=torch.arange(8,device=device)!=hot_slot
            records[mode].append({'step':step,'hot_correct':int((predicted[:,0]==truth[:,0]).sum()),'hot_n':batch,
              'cold_correct':int((predicted[:,1:]==truth[:,1:]).sum()),'cold_n':batch*7,
              'predicted_ids':predicted.cpu().tolist(),'target_ids':truth.cpu().tolist(),
              'non_hot_top_slot_change_l2':float((state[:,other]-initial[:,other]).reshape(batch,-1).norm(dim=-1).mean()),
              'cold_entity_correct':[int((predicted[:,e]==truth[:,e]).sum()) for e in range(1,8)]})
    probe(0)
    for step in range(1,max(CHECKPOINTS)+1):
        value=torch.randint(8,(batch,),generator=generator).to(device);entity=torch.zeros(batch,device=device,dtype=torch.long)
        truth[:,0]=value
        for mode in states:states[mode]=observe_with_gate(model,states[mode],entity,value,mode)
        if step in CHECKPOINTS:probe(step)
    if weights_hash(model)!=before:raise AssertionError('parameters changed')
    return {'seed':seed,'checkpoint':metadata,'weights_sha256':before,'weights_unchanged':True,
            'initialization':'same original soft primed state cloned for all arms',
            'read_rule':'original soft model.routing, original query network unchanged',
            'top_slots':top.cpu().tolist(),'hot_slot':hot_slot,
            'cold_entities_sharing_hot_top_slot':[e for e in range(1,8) if int(top[e])==hot_slot],
            'records':records,'batch_streams':batch}


def main():
    out=ROOT/'experiments/recurrent_state/write_gate8192.json'
    if out.exists():raise FileExistsError(out)
    report={'scope':'same fixed learned model and same writes, only hot-write gate changes; not trained sparse model',
            'provenance':capture(['models/state_write_intervention.py','models/slotted_working_state.py',
                                  'evaluation/write_gate_control.py','evaluation/working_state_checkpoint.py']),
            'runs':[]}
    for seed in (0,1,2):
        result=run(seed);report['runs'].append(result)
        print(json.dumps({'seed':seed,'final':{mode:{'hot':rows[-1]['hot_correct']/rows[-1]['hot_n'],
                    'cold':rows[-1]['cold_correct']/rows[-1]['cold_n'],'other_slot_drift':rows[-1]['non_hot_top_slot_change_l2']}
                     for mode,rows in result['records'].items()}}),flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
