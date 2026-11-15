"""Persisted text model in causal Runtime: correction/eviction and midstream restore.

Only confirmed feedback writes M; no online optimizer. Tiny exact template/explicit
slot task, familiar 8 entities, held question templates for probes. Not free-form NLP.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from evaluation.text_checkpoint import load
from evaluation.text_facts import ENTITIES,TEMPLATES,COLORS
from evaluation.provenance import capture
from runtime.controlled_text import ControlledTextSession
from runtime.model_session import state_hash
from runtime.session import StaleProposalError

ROOT=Path(__file__).resolve().parents[1]


def events(seed,count):
    gen=torch.Generator().manual_seed(seed+12000)
    out=[]
    for index in range(count):
        entity=int(torch.randint(0,8,(),generator=gen))
        truth=int(torch.randint(0,8,(),generator=gen))
        corrupt=index%12==0
        out.append({'entity':entity,'truth':COLORS[truth],
                    'feedback':COLORS[(truth+1)%8] if corrupt else COLORS[truth],
                    'corrupt':corrupt,'correction':False})
        if corrupt:
            out.append({'entity':entity,'truth':COLORS[truth],'feedback':COLORS[truth],
                        'corrupt':False,'correction':True})
    return out


def run_arm(checkpoint,device,sequence,capacity):
    model,task,metadata=load(checkpoint,device)
    fixed=state_hash(model)
    session=ControlledTextSession(model,task,capacity=capacity)
    clock=0; latest={}; traces=[]; restored_ok=False; stale_rejections=0; writes=0
    fixed_x=torch.stack([task.inputs[e,4] for e in range(8)])
    with torch.no_grad(): c_reference=model(fixed_x,1,sample_noise=False)['z_c'].clone()
    start=time.perf_counter()
    for index,event in enumerate(sequence):
        entity=event['entity']; question=TEMPLATES[0].format(entity=ENTITIES[entity])
        clock+=1; prediction=session.ask(question,event_time=clock)
        proposal=session.inner.propose(prediction['tokens_a'])
        text=json.dumps(prediction['tokens_a'])
        # First event exercises interruption, keeping already submitted prefix.
        if index==0:
            session.inner.commit(proposal,expected_offset=0,count=1)
            session.inner.interrupt('feedback_arriving')
            session.inner.resume(); session.inner.advance()
            try: session.inner.commit(proposal,expected_offset=1,count=1)
            except StaleProposalError: stale_rejections+=1
        else:
            session.inner.commit(proposal,expected_offset=0,count=len(text))
        memory_before=state_hash(session.inner.model,memory=True)
        clock+=1
        candidate=session.stage_feedback(prediction['observation_id'],event['feedback'],
                                          slot=f'entity:{entity}/color',event_time=clock)
        if state_hash(session.inner.model,memory=True)!=memory_before:
            raise AssertionError('Feedback staging wrote memory')
        receipt=session.confirm(candidate)
        if not session.confirm(candidate)['replayed']: raise AssertionError('duplicate write')
        writes+=receipt.get('write_calls',1)
        latest[entity]=event['truth']
        probes=[]
        for key in sorted(latest):
            clock+=1
            query=TEMPLATES[4+((index+key)%2)].format(entity=ENTITIES[key])
            result=session.ask(query,event_time=clock)
            probes.append({'entity':key,'question':query,'expected_latest_truth':latest[key],
                           'answer_a':result['answer_a'],'answer_c':result['answer_c'],
                           'correct':result['answer_a']==latest[key]})
        with torch.no_grad():
            c_now=session.inner.model(fixed_x,1,sample_noise=False)['z_c']
        if not torch.equal(c_reference,c_now): raise AssertionError('C changed during feedback writes')
        if state_hash(session.inner.model)!=fixed: raise AssertionError('weights changed')
        traces.append({**event,'event':index,'question':question,
                       'prediction_before_feedback':prediction['answer_a'],
                       'query_after_feedback':next(x['answer_a'] for x in probes if x['entity']==entity),
                       'probes':probes,'accuracy':sum(x['correct'] for x in probes)/len(probes),
                       'retained_slots':list(session.inner.controller.slots) if capacity else None,
                       'memory_version':receipt['version']})
        if index==len(sequence)//2:
            snapshot=json.loads(json.dumps(session.snapshot()))
            m2,t2,_=load(checkpoint,device)
            restored=ControlledTextSession.restore(m2,t2,snapshot)
            if session.snapshot()!=restored.snapshot(): raise AssertionError('snapshot mismatch')
            # Same next question on original and restored session, same timestamp.
            clock+=1
            query=TEMPLATES[5].format(entity=ENTITIES[entity])
            a=session.ask(query,event_time=clock); b=restored.ask(query,event_time=clock)
            if a!=b: raise AssertionError('restored text prediction diverged')
            session=restored; restored_ok=True
    corrections=[x for x in traces if x['correction']]
    return {'checkpoint_sha256':metadata['sha256'],'capacity':capacity,'events':traces,
            'seconds':time.perf_counter()-start,'average_accuracy':sum(x['accuracy'] for x in traces)/len(traces),
            'final_accuracy':traces[-1]['accuracy'],
            'corrective_feedback_post_accuracy':sum(x['query_after_feedback']==x['truth'] for x in corrections)/len(corrections),
            'numerical_write_calls':writes,'stale_commit_rejections':stale_rejections,
            'midstream_restore_prediction_equal':restored_ok,'parameters_unchanged':True,'c_bitwise_stable':True,
            'external_ledger':capacity is not None,
            'policy':'append' if capacity is None else 'latest-confirmed bounded LRU rebuild'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]); p.add_argument('--events',type=int,default=36)
    p.add_argument('--models',default='experiments/text_runtime/models800')
    p.add_argument('--output',default='experiments/text_runtime/stream36.json')
    args=p.parse_args(); out=ROOT/args.output
    if out.exists(): raise FileExistsError(out)
    if args.events<12:p.error('need >=12 events')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    report={'config':vars(args),'device':device,'scope':'explicit slots and restricted Chinese templates, not open conversation',
            'provenance':capture(['evaluation/text_runtime_stream.py','runtime/controlled_text.py',
                                  'runtime/bounded_session.py','runtime/bounded_memory.py','runtime/model_session.py']),
            'runs':[]}
    for seed in args.seeds:
        sequence=events(seed,args.events); arms={}
        for name,capacity in [('append',None),('bounded4',4),('bounded8',8)]:
            arms[name]=run_arm(ROOT/args.models/f'seed{seed}.pt',device,sequence,capacity)
            print(json.dumps({'seed':seed,'arm':name,**{k:v for k,v in arms[name].items() if k!='events'}}),flush=True)
        report['runs'].append({'seed':seed,'arms':arms})
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
