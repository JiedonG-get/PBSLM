import asyncio
import unittest
from runtime.continuous import ContinuousRunner
from runtime.dependency_router import DependencyRouter


class DependencyRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_disjoint_progress_then_relevant_update_interrupts(self):
        class Backend:
            def __init__(self):self.gates=asyncio.Queue();self.requests=[]
            async def generate(self,request):
                self.requests.append(request)
                for i in range(3):
                    await self.gates.get();yield f'{request.revision}:{i};'
        backend=Backend();runner=ContinuousRunner(backend);router=DependencyRouter(runner)
        try:
            initial=await runner.observe('甲号设备的颜色是什么？')
            await runner.wait_event(lambda e:e['kind']=='generation_started')
            router.set_dependencies({0})
            for i in range(20):
                await router.observe('乙号设备的颜色是蓝。')
            backend.gates.put_nowait(True)
            await runner.wait_event(lambda e:e['kind']=='output_committed' and e['generation_revision']==initial)
            prefix=runner.output
            correction=await router.observe('把甲号设备的颜色更新为绿。')
            await runner.wait_event(lambda e:e['kind']=='generation_started' and e['generation_revision']==correction)
            backend.gates.put_nowait(True)
            await runner.wait_event(lambda e:e['kind']=='output_committed' and e['generation_revision']==correction)
            self.assertTrue(runner.output.startswith(prefix))
            self.assertTrue(all(e['route']=='background' for e in router.journal[:-1]))
            self.assertEqual(router.journal[-1]['route'],'interrupt')
            self.assertIsNone(router.dependencies)
            self.assertEqual(sum(x['revision']==initial for x in runner.committed),1)
        finally:await runner.close()

    async def test_unknown_negated_question_and_no_dependencies_fail_closed(self):
        class Empty:
            async def generate(self,request):
                if False:yield ''
        runner=ContinuousRunner(Empty());router=DependencyRouter(runner)
        try:
            for text in ['乙号设备不是蓝色。','乙号设备的颜色是什么？','乙号设备也许是蓝色。','陌生电影画面']:
                router.set_dependencies({0})
                await router.observe(text,respond_on_interrupt=False)
                self.assertEqual(router.journal[-1]['route'],'interrupt')
            await router.observe('乙号设备的颜色是蓝。',respond_on_interrupt=False)
            self.assertEqual(router.journal[-1]['route'],'interrupt')
            with self.assertRaises(ValueError):router.set_dependencies(set())
        finally:await runner.close()


if __name__=='__main__':unittest.main()
