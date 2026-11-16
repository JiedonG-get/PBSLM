"""Actual fixed model rollout with explicitly packed event and committed-output tokens.

This checkpoint was trained for short fact questions, not this mixed context layout.
We validate transport and interruption only, never claim semantic continuation.
"""
import asyncio
from .event_context import EventContextPacker
from .pbslm_token_backend import PBSLMTokenBackend


class PBSLMContextBackend(PBSLMTokenBackend):
    def __init__(self,checkpoint,*,device='cpu',max_new_tokens=8,step_delay=.01,
                 history_tokens=3,prefix_tokens=3):
        super().__init__(checkpoint,device=device,max_new_tokens=max_new_tokens,step_delay=step_delay)
        self.packer=EventContextPacker(self.codec,history_tokens=history_tokens,
                                       prefix_tokens=prefix_tokens,min_latest_tokens=2)
        self.packing=[]

    def _context_step(self,request,context,step):
        result=self._step(request,context,step)
        with self._lock:
            self.metrics[-1]['actual_compact_input_ids']=list(context)
            self.metrics[-1]['step0_pack_applied']=step==0
            self.metrics[-1]['initial_event_pack_generation_revision']=request.revision
        return result

    async def generate(self,request):
        if self.closed:raise RuntimeError('backend closed')
        packed=self.packer.pack(request)
        self.packing.append({'revision':request.revision,'latest_revision':packed.latest_revision,
                             'initial_compact_ids':list(packed.compact_ids),
                             'spans':list(packed.spans),'truncation':list(packed.truncation),
                             'context_layout_was_not_trained':True})
        context=list(packed.compact_ids)
        loop=asyncio.get_running_loop()
        for step in range(self.max_new_tokens):
            token,surface=await loop.run_in_executor(self.executor,self._context_step,request,context,step)
            if self.step_delay:await asyncio.sleep(self.step_delay)
            yield surface
            context=(context+[token])[-self.task.length:]
