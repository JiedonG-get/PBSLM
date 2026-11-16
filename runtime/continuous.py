"""Event-loop continuous input and interruptible output, fixed-weight backends.

Protocol correctness, not language understanding. Backend yields atomic output units
(tokens/chars by its own definition). Cancellation is cooperative; revision fences
also reject output from a backend that ignores cancellation. No GPU preemption.
"""
import asyncio
from collections import deque
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class GenerationRequest:
    revision: int
    inputs: tuple
    committed_prefix: str
    started_at: float


class ContinuousRunner:
    def __init__(self, backend, *, history_limit=64, max_committed_chars=100000,
                 max_generation_tasks=4):
        if type(max_generation_tasks) is not int or max_generation_tasks<1:
            raise ValueError('positive generation task bound required')
        self.max_generation_tasks=max_generation_tasks
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError('positive history limit required')
        if type(max_committed_chars) is not int or max_committed_chars < 1:
            raise ValueError('positive output budget required')
        self.backend = backend
        self.inputs = deque(maxlen=history_limit)
        self.max_committed_chars = max_committed_chars
        self.revision = 0
        self.committed = []
        self.journal = []
        self.state = 'listening'
        self.closed = False
        self._active = None
        self._tasks = set()
        self._changed = asyncio.Event()
        self._started = time.perf_counter()

    @property
    def output(self):
        return ''.join(x['unit'] for x in self.committed)

    def _record(self, kind, **details):
        self.journal.append({'kind': kind, 'at': time.perf_counter()-self._started,
                             'revision': self.revision, **details})
        self._changed.set()

    def _invalidate(self, reason):
        if self._active is not None and not self._active.done():
            self._active.cancel()
            self._record('cancel_requested', reason=reason)
        self._active = None

    async def observe(self, text, *, should_respond=True):
        """Append event atomically before awaiting anything; never wait for inference."""
        if self.closed:
            raise RuntimeError('runner is closed')
        if not isinstance(text,str) or not text:
            raise ValueError('nonempty input text required')
        if type(should_respond) is not bool:
            raise ValueError('explicit boolean response policy required')
        # Optional synchronous queue submission: must not perform inference or await.
        # Queue rejection happens before revision/history changes, so callers can retry.
        if hasattr(self.backend,'submit_input'):
            self.backend.submit_input(self.revision+1,text)
        self.revision += 1
        self._invalidate('new_input')
        self.inputs.append((self.revision,text))
        self._record('input_received', text=text, should_respond=should_respond)
        self.state='listening'
        if should_respond:
            if sum(not task.done() for task in self._tasks)>=self.max_generation_tasks:
                self._record('response_deferred',reason='generation_task_capacity')
                return self.revision
            request=GenerationRequest(self.revision,tuple(self.inputs),self.output,
                                      time.perf_counter()-self._started)
            task=asyncio.create_task(self._consume(request))
            self._active=task;self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return self.revision

    async def interrupt(self, reason='external_interrupt'):
        if self.closed: raise RuntimeError('runner is closed')
        self.revision += 1
        self._invalidate(reason)
        self.state='listening'
        self._record('interrupted',reason=reason)

    async def resume(self):
        """Replan without inventing another input event or evicting real history."""
        if self.closed or not self.inputs:raise RuntimeError('open runner with input required')
        self.revision+=1
        self._invalidate('resume_replan')
        self._record('resume_requested')
        if sum(not task.done() for task in self._tasks)>=self.max_generation_tasks:
            self._record('response_deferred',reason='generation_task_capacity')
            return self.revision
        request=GenerationRequest(self.revision,tuple(self.inputs),self.output,
                                  time.perf_counter()-self._started)
        task=asyncio.create_task(self._consume(request))
        self._active=task;self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return self.revision

    async def _consume(self, request):
        self._record('generation_started', generation_revision=request.revision)
        stream=None
        try:
            stream=self.backend.generate(request)
            async for unit in stream:
                # No await between revision check and append: this is the commit fence.
                if self.closed or request.revision != self.revision:
                    self._record('stale_unit_dropped',generation_revision=request.revision)
                    break
                if not isinstance(unit,str) or not unit:
                    raise ValueError('backend output unit must be nonempty text')
                if len(self.output)+len(unit)>self.max_committed_chars:
                    raise ValueError('committed output budget exceeded')
                self.state='speaking'
                item={'unit':unit,'revision':request.revision,
                      'at':time.perf_counter()-self._started}
                self.committed.append(item)
                self._record('output_committed',unit=unit,generation_revision=request.revision)
            if request.revision==self.revision and not self.closed:
                self.state='listening';self._record('generation_finished',generation_revision=request.revision)
        except asyncio.CancelledError:
            self._record('generation_cancelled',generation_revision=request.revision)
            raise
        except Exception as exc:
            self._record('generation_failed',generation_revision=request.revision,
                         error=type(exc).__name__+': '+str(exc))
            if request.revision==self.revision:self.state='listening'
        finally:
            if stream is not None and hasattr(stream,'aclose'):
                try:
                    await stream.aclose()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._record('stream_cleanup_failed',generation_revision=request.revision,
                                 error=type(exc).__name__+': '+str(exc))

    async def wait_event(self, predicate, *, timeout=5):
        """Observe evidence; not a sleep/poll loop. Returns existing matching event too."""
        async def wait():
            while True:
                self._changed.clear()
                for event in self.journal:
                    if predicate(event):return event
                await self._changed.wait()
        return await asyncio.wait_for(wait(),timeout)

    async def close(self, *, timeout=5):
        if not self.closed:
            self.closed=True;self.revision+=1;self._invalidate('close');self.state='closed'
        for task in list(self._tasks):task.cancel()
        if self._tasks:
            done,pending=await asyncio.wait(list(self._tasks),timeout=timeout)
            for task in done:
                if not task.cancelled():task.exception()
            if pending:
                self._record('shutdown_timeout',pending_tasks=len(pending))
                raise TimeoutError('backend tasks ignored shutdown; not a hard realtime backend')
        self._record('closed')
