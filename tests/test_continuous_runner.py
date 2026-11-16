"""Deterministic continuous-Online protocol tests; no model or training.

Events gate backend progress. Timeouts only bound deadlocks; no timing sleeps are
used to assert concurrency. Units are protocol text units, not language quality.
"""
import asyncio
import unittest

from runtime.continuous import ContinuousRunner


class GatedBackend:
    def __init__(self):
        self.requests = []
        self.started = asyncio.Queue()
        self.release = {}
        self.cancelled = {}

    async def generate(self, request):
        revision = request.revision
        self.requests.append(request)
        self.release[revision] = asyncio.Event()
        self.cancelled[revision] = asyncio.Event()
        await self.started.put(request)
        yield f"r{revision}:prefix;"
        try:
            await self.release[revision].wait()
            yield f"r{revision}:suffix;"
        except asyncio.CancelledError:
            self.cancelled[revision].set()
            raise


class IgnoringOnceBackend:
    """First generation yields once after swallowing a cancellation."""
    def __init__(self):
        self.started = asyncio.Event()
        self.waiting = asyncio.Event()
        self.cancellation_seen = asyncio.Event()

    async def generate(self, request):
        self.started.set()
        yield 'kept-prefix;'
        self.waiting.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancellation_seen.set()
            yield 'STALE-MUST-NOT-COMMIT'


class ContinuousRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runners = []

    def runner(self, backend, **kwargs):
        runner = ContinuousRunner(backend, **kwargs)
        self.runners.append(runner)
        return runner

    async def asyncTearDown(self):
        for runner in self.runners:
            await runner.close(timeout=1)

    async def next_request(self, backend):
        return await asyncio.wait_for(backend.started.get(), timeout=1)

    async def committed(self, runner, revision):
        return await runner.wait_event(lambda e: e['kind'] == 'output_committed'
                                       and e['generation_revision'] == revision, timeout=1)

    async def test_input_while_output_active_preserves_prefix_and_replans(self):
        backend = GatedBackend()
        runner = self.runner(backend)
        first_revision = await runner.observe('movie scene one')
        first = await self.next_request(backend)
        await self.committed(runner, first_revision)
        prefix = runner.output
        self.assertEqual(runner.state, 'speaking')
        self.assertFalse(backend.release[first_revision].is_set())

        second_revision = await runner.observe('new scene: character changed plans')
        second = await self.next_request(backend)
        await self.committed(runner, second_revision)
        await asyncio.wait_for(backend.cancelled[first_revision].wait(), timeout=1)
        self.assertEqual(first.inputs, ((first_revision, 'movie scene one'),))
        self.assertEqual(second.inputs[-1], (second_revision, 'new scene: character changed plans'))
        self.assertEqual(second.committed_prefix, prefix)
        self.assertTrue(runner.output.startswith(prefix))
        self.assertNotIn(f'r{first_revision}:suffix;', runner.output)
        backend.release[second_revision].set()
        await runner.wait_event(lambda e: e['kind'] == 'generation_finished'
                                and e['generation_revision'] == second_revision, timeout=1)
        self.assertIn(f'r{second_revision}:suffix;', runner.output)

    async def test_revision_fence_drops_backend_that_ignores_one_cancel(self):
        backend = IgnoringOnceBackend()
        runner = self.runner(backend)
        revision = await runner.observe('first context')
        await asyncio.wait_for(backend.waiting.wait(), timeout=1)
        await self.committed(runner, revision)
        prefix = runner.output
        newer = await runner.observe('incoming correction', should_respond=False)
        await asyncio.wait_for(backend.cancellation_seen.wait(), timeout=1)
        event = await runner.wait_event(lambda e: e['kind'] == 'stale_unit_dropped'
                                       and e['generation_revision'] == revision, timeout=1)
        self.assertEqual(event['revision'], newer)
        self.assertEqual(runner.output, prefix)
        self.assertNotIn('STALE', runner.output)

    async def test_silent_policy_keeps_accepting_inputs_without_generation(self):
        backend = GatedBackend()
        runner = self.runner(backend)
        revision = await runner.observe('answer in progress')
        await self.next_request(backend)
        await self.committed(runner, revision)
        prefix = runner.output
        silent = await runner.observe('keep listening', should_respond=False)
        silent2 = await runner.observe('more incoming content', should_respond=False)
        await runner.wait_event(lambda e: e['kind'] == 'generation_cancelled'
                                and e['generation_revision'] == revision, timeout=1)
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(runner.output, prefix)
        self.assertEqual(runner.state, 'listening')
        self.assertEqual(list(runner.inputs)[-2:], [(silent, 'keep listening'), (silent2, 'more incoming content')])
        self.assertFalse(any(e['kind'] == 'generation_started' and e['generation_revision'] in (silent, silent2)
                             for e in runner.journal))

    async def test_repeated_interruptions_resume_latest_input_and_prefix(self):
        backend = GatedBackend()
        runner = self.runner(backend)
        revision = await runner.observe('latest observation')
        await self.next_request(backend)
        await self.committed(runner, revision)
        prefixes = []
        for turn in range(3):
            prefixes.append(runner.output)
            await runner.interrupt(f'interruption-{turn}')
            await runner.wait_event(lambda e: e['kind'] == 'generation_cancelled'
                                    and e['generation_revision'] == revision, timeout=1)
            self.assertEqual(runner.output, prefixes[-1])
            revision = await runner.resume()
            request = await self.next_request(backend)
            self.assertEqual(request.inputs[-1][1], 'latest observation')
            self.assertEqual(len(request.inputs),1)
            self.assertEqual(request.inputs[0],(1,'latest observation'))
            self.assertEqual(request.committed_prefix, prefixes[-1])
            await self.committed(runner, revision)
        self.assertEqual(sum(e['kind'] == 'interrupted' for e in runner.journal), 3)
        self.assertFalse(any(':suffix;' in item['unit'] for item in runner.committed))

    async def test_close_stops_future_commits_and_rejects_new_work(self):
        backend = GatedBackend()
        runner = self.runner(backend)
        revision = await runner.observe('input')
        await self.next_request(backend)
        await self.committed(runner, revision)
        prefix = runner.output
        await runner.close(timeout=1)
        backend.release[revision].set()
        self.assertEqual(runner.output, prefix)
        self.assertEqual(runner.state, 'closed')
        self.assertTrue(runner.closed)
        self.assertTrue(all(task.done() for task in runner._tasks))
        closed_index = next(i for i,e in enumerate(runner.journal) if e['kind'] == 'closed')
        self.assertFalse(any(e['kind'] == 'output_committed' for e in runner.journal[closed_index+1:]))
        for operation in (runner.observe('late'), runner.resume(), runner.interrupt()):
            with self.assertRaises(RuntimeError):
                await operation
        await runner.close(timeout=1)  # idempotent close

    async def test_backend_exception_logged_and_listening_recovers(self):
        class Failing:
            async def generate(self, request):
                yield 'prefix'
                raise LookupError('injected backend fault')
        runner = self.runner(Failing())
        revision = await runner.observe('question')
        event = await runner.wait_event(lambda e: e['kind'] == 'generation_failed', timeout=1)
        self.assertEqual(event['generation_revision'], revision)
        self.assertIn('LookupError: injected backend fault', event['error'])
        self.assertEqual(runner.output, 'prefix')
        self.assertEqual(runner.state, 'listening')

    async def test_history_bound_and_output_budget(self):
        class Recording:
            def __init__(self):
                self.requests = []
            async def generate(self, request):
                self.requests.append(request)
                yield 'abc'
                yield 'def'
        backend = Recording()
        runner = self.runner(backend, history_limit=2, max_committed_chars=5)
        await runner.observe('discarded old input', should_respond=False)
        second = await runner.observe('retained input', should_respond=False)
        third = await runner.observe('latest input')
        event = await runner.wait_event(lambda e: e['kind'] == 'generation_failed', timeout=1)
        self.assertIn('committed output budget exceeded', event['error'])
        self.assertEqual(runner.output, 'abc')
        self.assertEqual(backend.requests[0].inputs, ((second, 'retained input'), (third, 'latest input')))
        self.assertEqual(len(runner.inputs), 2)
        self.assertLessEqual(len(runner.output), 5)

    async def test_stream_finally_runs_on_budget_and_stale_drop(self):
        class Cleanup:
            def __init__(self):self.closed=asyncio.Event()
            async def generate(self,request):
                try:
                    yield 'over-budget'
                    await asyncio.Event().wait()
                finally:self.closed.set()
        backend=Cleanup();runner=self.runner(backend,max_committed_chars=2)
        await runner.observe('question')
        await runner.wait_event(lambda x:x['kind']=='generation_failed')
        await asyncio.wait_for(backend.closed.wait(),1)

    async def test_close_timeout_can_be_retried(self):
        class Resistant:
            def __init__(self):self.started=asyncio.Event();self.release=asyncio.Event();self.finished=asyncio.Event()
            async def generate(self,request):
                self.started.set()
                try:
                    while not self.release.is_set():
                        try:await self.release.wait()
                        except asyncio.CancelledError:pass
                    yield 'too-late'
                finally:self.finished.set()
        backend=Resistant();runner=self.runner(backend)
        await runner.observe('question');await backend.started.wait()
        with self.assertRaises(TimeoutError):await runner.close(timeout=0)
        backend.release.set()
        await runner.close(timeout=1)
        self.assertTrue(backend.finished.is_set());self.assertEqual(runner.output,'')
        self.assertTrue(any(x['kind']=='closed' for x in runner.journal))

    async def test_close_fence_for_cancellation_resistant_unit(self):
        backend = IgnoringOnceBackend()
        runner = self.runner(backend)
        revision = await runner.observe('input')
        await asyncio.wait_for(backend.waiting.wait(), timeout=1)
        await self.committed(runner, revision)
        prefix = runner.output
        await runner.close(timeout=1)
        self.assertEqual(runner.output, prefix)
        self.assertTrue(any(e['kind'] == 'stale_unit_dropped' for e in runner.journal))
        self.assertEqual(runner.state, 'closed')


if __name__ == '__main__':
    unittest.main()
