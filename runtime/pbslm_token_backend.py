"""Bounded neural token rollout for continuous scheduler experiments.

Existing tiny fact checkpoint is NOT trained for fluent AR generation. Rollout
quality is deliberately not scored as language; this backend tests real repeated
model steps, cancellation and scheduling only. It does not consume prior rendered
prefix semantically. New revision replans from its latest encoded input.
"""
import asyncio
import threading
import time

import torch
from .pbslm_backend import PBSLMQueryBackend


class PBSLMTokenBackend(PBSLMQueryBackend):
    def __init__(self,checkpoint,*,device='cpu',max_new_tokens=12,step_delay=.01):
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=128 or step_delay<0:
            raise ValueError('bounded positive token budget and nonnegative delay required')
        super().__init__(checkpoint,device=device,unit_delay=0)
        self.max_new_tokens=max_new_tokens;self.step_delay=step_delay

    def _step(self,request,context,step):
        start=time.perf_counter()
        with torch.inference_mode():
            ids=torch.tensor([context],device=self.model.tok.weight.device)
            out=self.model(ids,1,sample_noise=False)
            token=int(self.model.decode_nearest(out['z_a']).argmax(-1)[0,0])
            original_id=self.task.original_ids[token]
            surface=self.codec.tokenizer.decode([original_id],skip_special_tokens=False)
            if not surface:surface=f'<bpe:{original_id}>'
        with self._lock:
            self.metrics.append({'revision':request.revision,'step':step,'token':token,
                                 'surface':surface,'start_perf':start,'end_perf':time.perf_counter(),
                                 'thread':threading.current_thread().name})
        return token,surface

    async def generate(self,request):
        if self.closed:raise RuntimeError('backend closed')
        context=self.codec.encode(request.inputs[-1][1])
        loop=asyncio.get_running_loop()
        for step in range(self.max_new_tokens):
            token,surface=await loop.run_in_executor(self.executor,self._step,request,context,step)
            if self.step_delay:await asyncio.sleep(self.step_delay)
            yield surface
            # Actual generated token becomes next model context; fixed window contract.
            context=(context+[token])[-self.task.length:]
