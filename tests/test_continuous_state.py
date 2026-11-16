import asyncio
import threading
import unittest
from runtime.continuous import ContinuousRunner
from runtime.continuous_state import ContinuousStateBackend,ExactReferenceExtractor


class ContinuousStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_passive_ingest_processed_before_any_output(self):
        backend=ContinuousStateBackend(ExactReferenceExtractor(),unit_delay=0)
        runner=ContinuousRunner(backend)
        try:
            revision=await runner.observe('甲号设备的颜色是红。',should_respond=False)
            state=await asyncio.wait_for(backend.processed(revision),2)
            self.assertEqual(state[0]['color'],0)
            self.assertEqual(runner.output,'')
            query=await runner.observe('甲号设备的颜色是什么？')
            await runner.wait_event(lambda e:e['kind']=='generation_finished' and e['generation_revision']==query)
            self.assertIn('甲号设备的颜色记录为红。',runner.output)
        finally:
            await runner.close();await backend.aclose()

    async def test_output_cancel_does_not_cancel_ingest(self):
        entered=threading.Event();release=threading.Event();exact=ExactReferenceExtractor()
        def blocked(text):
            entered.set()
            if not release.wait(2):raise TimeoutError('test release')
            return exact(text)
        backend=ContinuousStateBackend(blocked,unit_delay=0);runner=ContinuousRunner(backend)
        try:
            first=await runner.observe('甲号设备的颜色是红。')
            self.assertTrue(await asyncio.to_thread(entered.wait,1))
            nextrev=await runner.observe('把甲号设备的颜色更新为绿。',should_respond=False)
            release.set()
            state=await asyncio.wait_for(backend.processed(nextrev),3)
            self.assertEqual(state[0]['color'],2)
            self.assertTrue(backend.futures[first].done())
            self.assertFalse(backend.futures[first].cancelled())
            self.assertEqual(runner.output,'')
        finally:
            release.set();await runner.close();await backend.aclose()

    async def test_queue_rejection_leaves_runner_history_unchanged(self):
        backend=ContinuousStateBackend(ExactReferenceExtractor(),queue_limit=1)
        runner=ContinuousRunner(backend)
        try:
            await runner.observe('甲号设备的颜色是红。',should_respond=False)
            before=(runner.revision,list(runner.inputs))
            with self.assertRaises(asyncio.QueueFull):
                await runner.observe('乙号设备的颜色是蓝。',should_respond=False)
            self.assertEqual(before,(runner.revision,list(runner.inputs)))
        finally:
            await runner.close();await backend.aclose()

    async def test_returned_state_copy_and_failed_snapshot_bound(self):
        backend=ContinuousStateBackend(ExactReferenceExtractor(),snapshot_limit=2,unit_delay=0)
        runner=ContinuousRunner(backend)
        try:
            revision=await runner.observe('甲号设备的颜色是红。',should_respond=False)
            state=await backend.processed(revision);state[0]['color']=7
            self.assertEqual((await backend.processed(revision))[0]['color'],0)
            def fail(text):raise RuntimeError('extract failed')
            backend.extractor=fail
            for i in range(3):
                r=await runner.observe('bad',should_respond=False)
                with self.assertRaisesRegex(RuntimeError,'extract failed'):await backend.processed(r)
            self.assertLessEqual(len(backend.futures),2)
            self.assertLessEqual(len(backend.states),2)
        finally:
            await runner.close();await backend.aclose()

    async def test_cancelled_close_can_resume_drain(self):
        entered=threading.Event();release=threading.Event()
        def blocked(text):
            entered.set()
            if not release.wait(3):raise TimeoutError('test release')
            return {'accepted':False}
        backend=ContinuousStateBackend(blocked);runner=ContinuousRunner(backend)
        try:
            r=await runner.observe('input',should_respond=False)
            self.assertTrue(await asyncio.to_thread(entered.wait,1))
            closer=asyncio.create_task(backend.aclose())
            await asyncio.sleep(0)  # schedule aclose; progress remains gated by threading.Event
            closer.cancel()
            with self.assertRaises(asyncio.CancelledError):await closer
            release.set()
            await asyncio.wait_for(backend.aclose(),2)
            self.assertTrue(backend.worker.done())
            await backend.processed(r)
        finally:
            release.set();await runner.close();await backend.aclose()

    async def test_worker_cancel_settles_accepted_inputs_and_close_reports(self):
        entered=threading.Event();release=threading.Event()
        def blocked(text):
            entered.set()
            if not release.wait(3):raise TimeoutError('test release')
            return {'accepted':False}
        backend=ContinuousStateBackend(blocked);runner=ContinuousRunner(backend)
        try:
            a=await runner.observe('first',should_respond=False)
            self.assertTrue(await asyncio.to_thread(entered.wait,1))
            b=await runner.observe('second',should_respond=False)
            backend.worker.cancel()
            with self.assertRaises(asyncio.CancelledError):await backend.worker
            for revision in (a,b):
                with self.assertRaisesRegex(RuntimeError,'worker stopped'):
                    await asyncio.wait_for(backend.processed(revision),1)
            with self.assertRaises(RuntimeError):await runner.observe('third',should_respond=False)
            release.set()
            with self.assertRaisesRegex(RuntimeError,'worker was cancelled'):
                await backend.aclose()
        finally:
            release.set();await runner.close()
            if backend._closing is None:
                try:await backend.aclose()
                except RuntimeError:pass

    async def test_unknown_or_ambiguous_input_does_not_mutate_belief(self):
        backend=ContinuousStateBackend(ExactReferenceExtractor(),state_capacity=1,unit_delay=0)
        runner=ContinuousRunner(backend)
        try:
            a=await runner.observe('甲号设备的颜色是红。',should_respond=False)
            await backend.processed(a)
            b=await runner.observe('甲号设备也许是蓝色。',should_respond=False)
            state=await backend.processed(b);self.assertEqual(state[0]['color'],0)
            c=await runner.observe('乙号设备的颜色是蓝。',should_respond=False)
            state=await backend.processed(c);self.assertEqual(set(state),{1})
        finally:
            await runner.close();await backend.aclose()


if __name__=='__main__':unittest.main()
