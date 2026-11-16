"""Explicit caller relevance, not a learned speech policy or semantic guarantee."""
import asyncio
import unittest
from runtime.continuous import ContinuousRunner
from runtime.continuous_state import ContinuousStateBackend,ExactReferenceExtractor


class BackgroundInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_background_burst_allows_current_output_to_progress(self):
        class Gated:
            def __init__(self):self.advance=asyncio.Queue();self.submitted=[]
            def submit_input(self,revision,text):self.submitted.append((revision,text))
            async def generate(self,request):
                for i in range(3):
                    await self.advance.get()
                    yield str(i)
        backend=Gated();runner=ContinuousRunner(backend)
        try:
            generation=await runner.observe('answer this')
            await runner.wait_event(lambda e:e['kind']=='generation_started')
            for i in range(100):
                await runner.observe_background(f'background{i}',relevance_assertion='test oracle: unrelated topic')
            for i in range(3):
                backend.advance.put_nowait(True)
                await runner.wait_event(lambda e:e['kind']=='output_committed' and e['unit']==str(i))
            await runner.wait_event(lambda e:e['kind']=='generation_finished')
            self.assertEqual(runner.output,'012');self.assertEqual(len(backend.submitted),101)
            self.assertEqual(runner.revision,101)
            for e in runner.journal:
                if e['kind']=='output_committed':
                    self.assertEqual(e['generation_revision'],generation)
                    self.assertEqual(e['output_fence_revision'],generation)
                    self.assertEqual(e['latest_observation_revision'],101)
        finally:await runner.close()

    async def test_hard_correction_still_invalidates_after_background(self):
        class Resistant:
            def __init__(self):self.started=asyncio.Event();self.release=asyncio.Event()
            async def generate(self,request):
                self.started.set()
                try:await self.release.wait()
                except asyncio.CancelledError:pass
                yield 'obsolete'
        backend=Resistant();runner=ContinuousRunner(backend)
        try:
            await runner.observe('old');await backend.started.wait()
            await runner.observe_background('unrelated',relevance_assertion='caller asserted unrelated')
            await runner.observe('correction',should_respond=False)
            backend.release.set()
            await runner.wait_event(lambda e:e['kind']=='stale_unit_dropped')
            self.assertEqual(runner.output,'')
        finally:
            backend.release.set();await runner.close()

    async def test_deferred_question_not_replaced_by_background(self):
        class Backend:
            def __init__(self):self.started=asyncio.Event();self.release=asyncio.Event();self.requests=[]
            async def generate(self,request):
                self.requests.append(request)
                if len(self.requests)==1:
                    self.started.set()
                    while not self.release.is_set():
                        try:await self.release.wait()
                        except asyncio.CancelledError:pass
                    yield 'stale'
                else:yield request.inputs[-1][1]
        backend=Backend();runner=ContinuousRunner(backend,max_generation_tasks=1)
        try:
            await runner.observe('old');await backend.started.wait()
            await runner.observe('pending question')
            revision=await runner.observe_background('unrelated fact',relevance_assertion='caller unrelated')
            backend.release.set()
            await runner.wait_event(lambda e:e['kind']=='generation_finished' and e['generation_revision']==revision)
            self.assertEqual(runner.output,'pending question')
            self.assertEqual(runner.inputs[-1][1],'unrelated fact')
            self.assertEqual(backend.requests[-1].inputs[-1][1],'pending question')
        finally:
            backend.release.set();await runner.close()

    async def test_background_updates_independent_ingest_state(self):
        backend=ContinuousStateBackend(ExactReferenceExtractor(),unit_delay=0);runner=ContinuousRunner(backend)
        try:
            rev=await runner.observe_background('乙号设备的颜色是蓝。',relevance_assertion='unrelated entity')
            self.assertEqual((await backend.processed(rev))[1]['color'],1)
            self.assertEqual(runner.output,'')
            with self.assertRaises(ValueError):await runner.observe_background('input',relevance_assertion='')
        finally:
            await runner.close();await backend.aclose()


if __name__=='__main__':unittest.main()
