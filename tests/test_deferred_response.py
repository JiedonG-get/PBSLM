import asyncio
import unittest
from runtime.continuous import ContinuousRunner


class DeferredResponseTests(unittest.IsolatedAsyncioTestCase):
    async def scenario(self,mode):
        class ResistantFirst:
            def __init__(self):self.started=asyncio.Event();self.release=asyncio.Event();self.requests=[]
            async def generate(self,request):
                self.requests.append(request)
                if len(self.requests)==1:
                    self.started.set()
                    while not self.release.is_set():
                        try:await self.release.wait()
                        except asyncio.CancelledError:pass
                    yield 'STALE'
                else:yield request.inputs[-1][1]
        backend=ResistantFirst();runner=ContinuousRunner(backend,max_generation_tasks=1)
        try:
            await runner.observe('old');await backend.started.wait()
            await runner.observe('middle')
            latest=await runner.observe('newest')
            self.assertEqual(len(backend.requests),1)
            if mode=='listen':await runner.observe('listen-only',should_respond=False)
            elif mode=='interrupt':await runner.interrupt()
            backend.release.set()
            await runner.wait_event(lambda e:e['kind']=='stale_unit_dropped')
            if mode=='answer':
                await runner.wait_event(lambda e:e['kind']=='generation_finished' and e['generation_revision']==latest)
                self.assertEqual(runner.output,'newest')
                self.assertEqual(len(backend.requests),2)
                self.assertEqual(backend.requests[-1].inputs[-1][1],'newest')
            else:
                await runner.close()
                self.assertEqual(runner.output,'');self.assertEqual(len(backend.requests),1)
        finally:
            backend.release.set();await runner.close()

    async def test_latest_pending_response_automatically_runs(self):await self.scenario('answer')
    async def test_silent_input_clears_pending_speech(self):await self.scenario('listen')
    async def test_explicit_interrupt_clears_pending_speech(self):await self.scenario('interrupt')


if __name__=='__main__':unittest.main()
