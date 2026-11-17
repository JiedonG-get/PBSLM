"""All hot IDs on same frozen soft-trained model: collision-aware gate diagnostic.

No training. Same original-soft priming and original-soft read for each arm.
Different hot IDs have dedicated fixed streams; compare paired modes within hotID.
"""
import argparse
import json
from pathlib import Path
import torch
from models.state_write_intervention import observe_with_gate
from evaluation.working_state_checkpoint import load_working_state
from evaluation.recurrent_state_control import weights_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(seed,args):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,metadata=load_working_state(ROOT/f'experiments/recurrent_state/slots600/learned_seed{seed}.pt',device)
    before=weights_hash(model);top=model.routing(torch.arange(8,device=device)).argmax(-1).tolist();cases=[]
    checkpoints=sorted({0,1,32,128,512,args.steps})
    if args.steps<512:raise ValueError('steps must be >=512')
    for hot in range(8):
        gen=torch.Generator().manual_seed(seed+131000+hot*100)
        truth=torch.randint(8,(args.batch,8),generator=gen).to(device);initial=model.initial(args.batch)
        for e in range(8):initial=model.observe(initial,torch.full((args.batch,),e,device=device,dtype=torch.long),truth[:,e])
        states={mode:initial.clone() for mode in ('soft','top1_mass')};records={mode:[] for mode in states}
        cold=[e for e in range(8) if e!=hot]
        colliding=[e for e in cold if top[e]==top[hot]]
        def probe(step):
            for mode,state in states.items():
                saved=state.clone()
                predicted=torch.stack([model.query(state,torch.full((args.batch,),e,device=device,dtype=torch.long)).argmax(-1) for e in range(8)],1)
                if not torch.equal(saved,state):raise AssertionError('query mutated state')
                hits=(predicted==truth).sum(0).cpu().tolist()
                records[mode].append({'step':step,'per_entity_correct':hits,'per_entity_n':args.batch,
                         'hot_correct':hits[hot],'cold_correct':sum(hits[e] for e in cold),'cold_n':args.batch*7,
                         'predicted_ids':predicted.cpu().tolist(),'target_ids':truth.cpu().tolist()})
        probe(0)
        for step in range(1,args.steps+1):
            values=torch.randint(8,(args.batch,),generator=gen).to(device)
            truth[:,hot]=values;entities=torch.full((args.batch,),hot,device=device,dtype=torch.long)
            for mode in states:states[mode]=observe_with_gate(model,states[mode],entities,values,mode)
            if step in checkpoints:probe(step)
        cases.append({'hot_entity':hot,'hot_top_slot':top[hot],'colliding_cold_entities':colliding,'records':records})
        print(json.dumps({'seed':seed,'hot':hot,'colliding':colliding,'final':{
          mode:{'hot':r[-1]['hot_correct']/args.batch,'cold':r[-1]['cold_correct']/r[-1]['cold_n']}
          for mode,r in records.items()}}),flush=True)
    if weights_hash(model)!=before:raise AssertionError('weights changed')
    return {'seed':seed,'checkpoint':metadata,'top_slots':top,'weights_sha256':before,'weights_unchanged':True,'cases':cases}


def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=2048);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--output',default='experiments/recurrent_state/all_hot2048.json');args=p.parse_args()
    if args.steps<512 or args.batch<1:p.error('steps>=512 and positive batch required')
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    report={'config':vars(args),'scope':'frozen-model test-time gate intervention, all known hot IDs, not trained sparse routing',
            'provenance':capture(['evaluation/all_hot_gate_control.py','models/state_write_intervention.py',
                        'models/slotted_working_state.py','evaluation/working_state_checkpoint.py']), 'runs':[]}
    for seed in (0,1,2):report['runs'].append(run(seed,args))
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
