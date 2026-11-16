"""Fixed PBSLM color-query backend, dedicated serial inference thread.

Actual neural inference returns a single color-token prediction. Pacing yielded IDs
is NOT autoregressive language generation. Runner keeps full recent event context,
but this small checkpoint consumes only latest question; no long-state understanding.
The worker is serialized even if asyncio futures are cancelled; running CPU/GPU
compute is not preempted. Metrics expose start/end on the inference thread.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import torch
from evaluation.text_checkpoint import load
from runtime.text_codec import CompactTextCodec
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class PBSLMQueryBackend:
    def __init__(self, checkpoint, *, device='cpu', unit_delay=.01):
        if unit_delay<0:raise ValueError('nonnegative delay required')
        self.model,self.task,self.artifact=load(checkpoint,device)
        self.codec=CompactTextCodec(self.task.original_ids,self.task.length,
                                    ROOT/'tokenizer/bpe16k_tokenizer.json',self.task.tokenizer_hash)
        self.unit_delay=unit_delay
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pbslm-inference')
        self.metrics=[];self._lock=threading.Lock();self.closed=False

    def _infer(self, request, encoded):
        start=time.perf_counter()
        with torch.inference_mode():
            ids=torch.tensor([encoded],device=self.model.tok.weight.device)
            out=self.model(ids,1,sample_noise=False)
            token=int(self.model.decode_nearest(out['z_a']).argmax(-1)[0,0])
            answer=self.task.color_tokens.get(token,f'<compact-token:{token}>')
        end=time.perf_counter()
        with self._lock:
            self.metrics.append({'revision':request.revision,'inference_started_perf':start,
                                 'inference_finished_perf':end,'seconds':end-start,
                                 'thread':threading.current_thread().name,
                                 'context_events_received':len(request.inputs),
                                 'committed_prefix_chars_received':len(request.committed_prefix),
                                 'model_consumes_only_latest_question':True,
                                 'token':token,'answer':answer})
        return answer

    async def generate(self,request):
        if self.closed:raise RuntimeError('backend closed')
        encoded=self.codec.encode(request.inputs[-1][1])
        loop=asyncio.get_running_loop()
        answer=await loop.run_in_executor(self.executor,self._infer,request,encoded)
        # Atomic unit is the entire one-token answer; renderer pacing is explicit.
        if self.unit_delay:await asyncio.sleep(self.unit_delay)
        yield answer

    def shutdown(self):
        self.closed=True
        self.executor.shutdown(wait=True,cancel_futures=True)
