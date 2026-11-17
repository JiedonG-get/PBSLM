"""Structured-event continuous execution of frozen learned fast state.

Independent diagnostic adapter, NOT integrated original PBSLM main model. JSON
writes contain continuous key+observed value; query contains key only. One serial
worker owns one matrix state; saved results (not raw history) serve output requests.
No external key/value retrieval or truth store, no parameter updates.
"""
import asyncio
import hashlib
import json
import math
from pathlib import Path
import torch
from .continuous_state import ContinuousStateBackend
from models.content_fast_state import ContentFastState


def parse_fast_event(text):
    event=json.loads(text)
    if not isinstance(event,dict) or event.get('op') not in ('write','query'):
        raise ValueError('structured write/query event required')
    expected={'op','key','value'} if event['op']=='write' else {'op','key'}
    if set(event)!=expected:raise ValueError('unexpected/missing event fields')
    key=event['key']
    if (not isinstance(key,list) or len(key)!=16 or any(isinstance(x,bool) or not isinstance(x,(int,float))
        or not math.isfinite(x) for x in key) or sum(x*x for x in key)<1e-12):
        raise ValueError('finite nonzero16D key required')
    tensor=torch.tensor(key,dtype=torch.float32)
    if not torch.isfinite(tensor).all() or not torch.isfinite(tensor.norm()) or tensor.norm()<1e-6:
        raise ValueError('key outside finite fp32 contract')
    if event['op']=='write' and (type(event['value']) is not int or not 0<=event['value']<8):
        raise ValueError('write value integer0..7 required')
    return event


def state_digest(state):return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


def load_fast_model(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if (ck.get('format')!='pbslm.content_fast_state.research.v1' or ck.get('rule')!='delta'
        or ck.get('config')!={'key_dim':16,'values':8}):raise ValueError('canonical delta checkpoint required')
    if any(not torch.isfinite(value).all() for value in ck['state_dict'].values()):raise ValueError('finite weights required')
    model=ContentFastState().eval();model.load_state_dict(ck['state_dict'],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    return model,{'checkpoint_file_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'seed':ck['seed']}


class FastStateEngine:
    def __init__(self,model):
        if model.training or model.rule!='delta' or next(model.parameters()).device.type!='cpu':
            raise ValueError('CPU eval delta model required')
        for p in model.parameters():p.requires_grad_(False)
        self.model=model;self.state=model.initial(1)

    @torch.inference_mode()
    def __call__(self,text):
        event=parse_fast_event(text);key=torch.tensor([event['key']],dtype=torch.float32)
        before=state_digest(self.state)
        if event['op']=='write':
            candidate=self.model.observe(self.state,key,torch.tensor([event['value']]))
            if not torch.isfinite(candidate).all():raise FloatingPointError('nonfinite candidate state')
            self.state=candidate
            result={'op':'write','prediction':None}
        else:
            scores=self.model.query(self.state,key)
            if not torch.isfinite(scores).all():raise FloatingPointError('nonfinite query logits')
            result={'op':'query','prediction':int(scores.argmax(-1)[0]),'probabilities':scores.softmax(-1)[0].tolist()}
        after=state_digest(self.state)
        if event['op']=='query' and before!=after:raise AssertionError('query changed state')
        return {**result,'state_before_sha256':before,'state_after_sha256':after,
                'state_floats':self.state.numel(),'state_bytes':self.state.numel()*self.state.element_size()}


class FastStateOnlineBackend(ContinuousStateBackend):
    def __init__(self,model,*,queue_limit=16,result_limit=16,output_gate=None):
        self.engine=FastStateEngine(model);self.output_gate=output_gate
        super().__init__(self.engine,queue_limit=queue_limit,snapshot_limit=result_limit,unit_delay=0)

    def submit_input(self,revision,text):
        parse_fast_event(text) # invalid events must not consume runner revision
        super().submit_input(revision,text)

    async def _work_loop(self):
        while True:
            item=await self.queue.get()
            if item is None:self.queue.task_done();return
            revision,text=item
            try:
                self.record('fast_event_started',revision=revision)
                result=await asyncio.get_running_loop().run_in_executor(self.executor,self.engine,text)
                self.record('fast_event_finished',revision=revision,result=result)
            except Exception as exc:
                result={'_error':type(exc).__name__+': '+str(exc)}
                self.record('fast_event_failed',revision=revision,error=result['_error'])
            finally:self.queue.task_done()
            self.states[revision]=result
            while len(self.states)>self.snapshot_limit:
                old,_=self.states.popitem(last=False);self.futures.pop(old,None)
            self.futures[revision].set_result(result)

    async def generate(self,request):
        result=await self.processed(request.inputs[-1][0])
        if result['op']!='query':return
        yield 'query_started;'
        if self.output_gate is not None:await self.output_gate.wait()
        yield json.dumps({'prediction':result['prediction'], 'unknown':result['prediction']==8,
                          'state_sha256':result['state_after_sha256']})
