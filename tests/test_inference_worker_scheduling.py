"""Deterministic worker scheduling tests using actual backend executor/generate path.

Model math replaced with thread-gated work: proves execution lifecycle, not neural
understanding. Thread Events, no sleep-based pass/fail or training/checkpoint needed.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from types import SimpleNamespace

from runtime.continuous import ContinuousRunner
from runtime.pbslm_backend import PBSLMQueryBackend


class WorkerSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_future_does_not_parallelize_or_commit_old_work(self):
        backend=PBSLMQueryBackend.__new__(PBSLMQueryBackend)
        backend.closed=False;backend.unit_delay=0
        backend.codec=SimpleNamespace(encode=lambda text:[1])
        backend.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='test-model-worker')
        loop=asyncio.get_running_loop()
        first_started=asyncio.Event();second_started=asyncio.Event()
        first_release=threading.Event();second_release=threading.Event()
        active=0;peak=0;order=[];lock=threading.Lock()

        def infer(request,encoded):
            nonlocal active,peak
            with lock:
                active+=1;peak=max(peak,active);order.append(('start',request.revision))
            try:
                if request.revision==1:
                    loop.call_soon_threadsafe(first_started.set)
                    if not first_release.wait(3):raise TimeoutError('test first release')
                else:
                    loop.call_soon_threadsafe(second_started.set)
                    if not second_release.wait(3):raise TimeoutError('test second release')
                return 'obsolete' if request.revision==1 else 'latest'
            finally:
                with lock:active-=1;order.append(('end',request.revision))

        backend._infer=infer
        runner=ContinuousRunner(backend)
        try:
            await runner.observe('first')
            await asyncio.wait_for(first_started.wait(),1)
            newer=await runner.observe('new information')
            # Thread blocked on first_release, yet observe returned and revision changed.
            self.assertEqual(newer,2);self.assertFalse(first_release.is_set())
            await runner.wait_event(lambda e:e['kind']=='generation_started' and e['generation_revision']==2)
            self.assertFalse(second_started.is_set())
            first_release.set()
            await asyncio.wait_for(second_started.wait(),1)
            self.assertEqual(order[:3],[('start',1),('end',1),('start',2)])
            second_release.set()
            await runner.wait_event(lambda e:e['kind']=='generation_finished' and e['generation_revision']==2)
            self.assertEqual(runner.output,'latest');self.assertEqual(peak,1)
            self.assertTrue(all(x['revision']==2 for x in runner.committed))
        finally:
            first_release.set();second_release.set()
            try:await runner.close()
            finally:await asyncio.to_thread(backend.shutdown)

    async def test_generation_capacity_defers_without_losing_input(self):
        class Resistant:
            def __init__(self):self.started=asyncio.Event();self.release=asyncio.Event()
            async def generate(self,request):
                self.started.set()
                while not self.release.is_set():
                    try:await self.release.wait()
                    except asyncio.CancelledError:pass
                yield 'stale'
        backend=Resistant();runner=ContinuousRunner(backend,max_generation_tasks=1)
        try:
            await runner.observe('old');await backend.started.wait()
            revision=await runner.observe('new')
            self.assertEqual(list(runner.inputs)[-1],(revision,'new'))
            self.assertEqual(len(runner._tasks),1)
            self.assertTrue(any(e['kind']=='response_deferred' for e in runner.journal))
            backend.release.set()
            await runner.wait_event(lambda e:e['kind']=='stale_unit_dropped')
            self.assertEqual(runner.output,'')
        finally:
            backend.release.set();await runner.close()


if __name__=='__main__':unittest.main()
