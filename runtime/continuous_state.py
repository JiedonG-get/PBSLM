"""Independent continuous ingest + symbolic belief-state reference backend.

Fixed extractors run serially off the event loop, even for should_respond=False.
State is caller-asserted extracted observations, NOT verified facts. Output wording
is rule-rendered, not PBSLM generation. LearnedReaderExtractor is a separate frozen
GRU; it proposes risky candidates, never writes model M or updates weights.
"""
import asyncio
import copy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import re
import time

from evaluation.text_facts import ENTITIES,COLORS


class ExactReferenceExtractor:
    """Known-syntax state baseline, deliberately not open language understanding."""
    def __call__(self,text):
        for entity,e in enumerate(ENTITIES):
            for color,c in enumerate(COLORS):
                if text in (f'{e}的颜色是{c}。',f'把{e}的颜色更新为{c}。'):
                    return {'accepted':True,'entity':entity,'color':color,'method':'exact_reference'}
        return {'accepted':False,'method':'exact_reference'}


class LearnedReaderExtractor:
    def __init__(self,path,device='cpu'):
        from .fact_reader import load_reader
        self.model,self.tokenizer,self.metadata=load_reader(path,device)
        self.device=device

    def __call__(self,text):
        from .fact_reader import infer
        return infer(self.model,[text],self.tokenizer,self.device,
                     self.metadata['threshold'],self.metadata['margin'])[0]


class ContinuousStateBackend:
    def __init__(self,extractor,*,queue_limit=32,state_capacity=16,snapshot_limit=64,unit_delay=.01):
        if any(type(v) is not int or v<1 for v in (queue_limit,state_capacity,snapshot_limit)) or unit_delay<0:
            raise ValueError('positive capacities and nonnegative delay required')
        self.extractor=extractor;self.queue=asyncio.Queue(maxsize=queue_limit)
        self.capacity=state_capacity;self.snapshot_limit=snapshot_limit;self.unit_delay=unit_delay
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pbslm-state-ingest')
        self.states=OrderedDict();self.futures={};self.beliefs=OrderedDict();self.journal=[]
        self.worker=None;self.closed=False;self._closing=None;self.last_input_revision=0
        self.origin=time.perf_counter()

    def record(self,kind,**details):
        self.journal.append(copy.deepcopy({'kind':kind,'at':time.perf_counter()-self.origin,**details}))

    def submit_input(self,revision,text):
        if self.closed or (self.worker is not None and self.worker.done()):
            raise RuntimeError('state backend closed or worker stopped')
        if revision<=self.last_input_revision:raise ValueError('input revisions must increase')
        # Validate capacity before modifying tracking structures.
        self.queue.put_nowait((revision,text))
        self.futures[revision]=asyncio.get_running_loop().create_future()
        self.last_input_revision=revision
        self.record('ingest_enqueued',revision=revision,text=text)
        if self.worker is None:self.worker=asyncio.create_task(self._work())

    async def _work(self):
        try:
            await self._work_loop()
        finally:
            # A failed/cancelled private worker must not strand accepted input futures.
            for revision,future in list(self.futures.items()):
                if not future.done():future.set_result({'_error':'ingest worker stopped before processing completed'})
            while not self.queue.empty():
                self.queue.get_nowait();self.queue.task_done()

    async def _work_loop(self):
        while True:
            item=await self.queue.get()
            if item is None:
                self.queue.task_done();return
            revision,text=item
            try:
                self.record('understanding_started',revision=revision)
                result=await asyncio.get_running_loop().run_in_executor(self.executor,self.extractor,text)
                if result.get('accepted'):
                    e,c=result['entity'],result['color']
                    if type(e) is not int or type(c) is not int or not 0<=e<len(ENTITIES) or not 0<=c<len(COLORS):
                        raise ValueError('invalid extracted entity/color')
                    self.beliefs.pop(e,None)
                    self.beliefs[e]={'color':c,'source_text':text,'input_revision':revision}
                    while len(self.beliefs)>self.capacity:self.beliefs.popitem(last=False)
                state={k:dict(v) for k,v in self.beliefs.items()}
                self.states[revision]=state
                while len(self.states)>self.snapshot_limit:
                    old,_=self.states.popitem(last=False);self.futures.pop(old,None)
                self.record('understanding_finished',revision=revision,extraction=result,
                            state={str(k):v for k,v in state.items()})
                self.futures[revision].set_result(state)
            except Exception as exc:
                self.record('understanding_failed',revision=revision,error=type(exc).__name__+': '+str(exc))
                # Return error as data so unused passive futures have no unhandled exception.
                error={'_error':type(exc).__name__+': '+str(exc)}
                self.states[revision]=error
                while len(self.states)>self.snapshot_limit:
                    old,_=self.states.popitem(last=False);self.futures.pop(old,None)
                self.futures[revision].set_result(error)
            finally:self.queue.task_done()

    async def processed(self,revision):
        future=self.futures.get(revision)
        if future is None:raise ValueError('input revision unknown or evicted')
        value=await asyncio.shield(future) # cancelled speaking must not cancel understanding
        if '_error' in value:raise RuntimeError(value['_error'])
        return copy.deepcopy(value)

    async def generate(self,request):
        input_revision,text=request.inputs[-1]
        state=await self.processed(input_revision)
        entities=[i for i,name in enumerate(ENTITIES) if name in text]
        if len(entities)!=1:
            self.record('response_abstained',revision=request.revision,reason='no_single_known_entity');return
        entity=entities[0]
        if entity not in state:
            answer=f'尚无{ENTITIES[entity]}的已接收记录。'
        else:answer=f'{ENTITIES[entity]}的颜色记录为{COLORS[state[entity]["color"]]}。'
        # This is explicit rule policy/renderer. New state does not rewrite spoken prefix.
        intro='接着已提交的内容，按最新记录：' if request.committed_prefix else '按当前收到的记录：'
        self.record('response_planned',generation_revision=request.revision,input_revision=input_revision,
                    answer=answer,prefix_chars=len(request.committed_prefix))
        yield intro
        if self.unit_delay:await asyncio.sleep(self.unit_delay)
        yield answer

    async def _drain(self):
        try:
            await self.queue.join()
            if self.worker:
                if not self.worker.done():await self.queue.put(None)
                try:await self.worker
                except asyncio.CancelledError as exc:
                    raise RuntimeError('ingest worker was cancelled') from exc
        finally:
            await asyncio.to_thread(self.executor.shutdown,wait=True,cancel_futures=True)
            self.record('backend_closed')

    async def aclose(self):
        # Caller timeout/cancellation doesn't cancel the actual drain operation.
        # Retry awaits the same task; running native work cannot be preempted.
        self.closed=True
        if self._closing is None:self._closing=asyncio.create_task(self._drain())
        await asyncio.shield(self._closing)
