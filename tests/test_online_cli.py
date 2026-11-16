import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from runtime.continuous import ContinuousRunner


class OnlineInterfaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_journal_cursor_has_no_replay_and_copies(self):
        class Empty:
            async def generate(self,r):
                yield 'hello'
        runner=ContinuousRunner(Empty())
        try:
            await runner.observe('input')
            await runner.wait_event(lambda e:e['kind']=='generation_finished')
            cursor,events=await runner.read_events(0)
            self.assertGreater(cursor,0)
            events[0]['kind']='mutated'
            self.assertNotEqual(runner.journal[0]['kind'],'mutated')
            await runner.observe('listen',should_respond=False)
            next_cursor,batch=await runner.read_events(cursor)
            self.assertEqual(len(batch),1);self.assertEqual(batch[0]['kind'],'input_received')
            self.assertGreater(next_cursor,cursor)
        finally:await runner.close()


class OnlineCLITests(unittest.TestCase):
    def test_piped_reference_state_no_checkpoint_required(self):
        with tempfile.TemporaryDirectory() as directory:
            trace=Path(directory)/'trace.json'
            program='/listen 甲号设备的颜色是红。\n/wait\n甲号设备的颜色是什么？\n/wait\n/quit\n'
            env={**os.environ,'PYTHONIOENCODING':'utf-8'}
            run=subprocess.run([sys.executable,'-m','runtime.online_cli','--delay','0', '--trace',str(trace)],
                               input=program,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               encoding='utf-8',env=env,timeout=20)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn('甲号设备的颜色记录为红。',run.stdout)
            self.assertTrue(trace.exists())
            payload=json.loads(trace.read_text(encoding='utf-8'))
            inputs=[e for e in payload['runtime'] if e['kind']=='input_received']
            self.assertEqual(len(inputs),2)
            self.assertTrue(all('/wait' not in e['text'] and '/quit' not in e['text'] for e in inputs))
            self.assertTrue(any(e['kind']=='closed' for e in payload['runtime']))


if __name__=='__main__':unittest.main()
