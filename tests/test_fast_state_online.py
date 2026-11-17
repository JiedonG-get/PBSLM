import asyncio
import json
import unittest
from unittest.mock import patch
import torch
from models.content_fast_state import ContentFastState
from runtime.continuous import ContinuousRunner
from runtime.fast_state_online import FastStateOnlineBackend,parse_fast_event


def event(op,key,value=None):
    row={'op':op,'key':key.tolist()}
    if op=='write':row['value']=value
    return json.dumps(row)


class FastStateOnlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_passive_write_query_cancel_and_correction(self):
        torch.manual_seed(4);model=ContentFastState().eval();gate=asyncio.Event()
        backend=FastStateOnlineBackend(model,output_gate=gate);runner=ContinuousRunner(backend)
        key=torch.eye(16)[0]
        try:
            write=await runner.observe(event('write',key,1),should_respond=False)
            a=await backend.processed(write);self.assertNotEqual(a['state_before_sha256'],a['state_after_sha256'])
            self.assertEqual(runner.output,'')
            oldquery=await runner.observe(event('query',key))
            await runner.wait_event(lambda e:e['kind']=='output_committed' and e['generation_revision']==oldquery)
            correction=await runner.observe(event('write',key,2),should_respond=False)
            await backend.processed(correction)
            newquery=await runner.observe(event('query',key))
            await runner.wait_event(lambda e:e['kind']=='output_committed' and e['generation_revision']==newquery)
            gate.set()
            await runner.wait_event(lambda e:e['kind']=='generation_finished' and e['generation_revision']==newquery)
            olditems=[x for x in runner.committed if x['revision']==oldquery]
            self.assertEqual(len(olditems),1);self.assertEqual(olditems[0]['unit'],'query_started;')
            result=await backend.processed(newquery)
            self.assertEqual(result['state_before_sha256'],result['state_after_sha256'])
            self.assertEqual(result['state_bytes'],512)
            current=await backend.processed(correction)
            self.assertEqual(result['state_before_sha256'],current['state_after_sha256'])
        finally:
            gate.set()
            try:await runner.close()
            finally:await backend.aclose()

    async def test_failed_write_preserves_matrix_and_worker_continues(self):
        model=ContentFastState().eval();backend=FastStateOnlineBackend(model);runner=ContinuousRunner(backend)
        key=torch.eye(16)[0]
        try:
            r=await runner.observe(event('write',key,1),should_respond=False)
            await backend.processed(r);saved=backend.engine.state.clone()
            with patch.object(model,'observe',side_effect=RuntimeError('injected write failure')):
                r=await runner.observe(event('write',key,2),should_respond=False)
                with self.assertRaisesRegex(RuntimeError,'injected write failure'):await backend.processed(r)
            self.assertTrue(torch.equal(saved,backend.engine.state))
            r=await runner.observe(event('query',key),should_respond=False)
            answer=await backend.processed(r);self.assertEqual(answer['op'],'query')
        finally:
            await runner.close();await backend.aclose()

    async def test_invalid_event_does_not_mutate_admission_and_result_bound(self):
        backend=FastStateOnlineBackend(ContentFastState().eval(),result_limit=2)
        runner=ContinuousRunner(backend)
        try:
            with self.assertRaises(ValueError):await runner.observe('{"op":"query","key":[0]}')
            self.assertEqual(runner.revision,0);self.assertEqual(backend.last_input_revision,0)
            first=None
            for i in range(4):
                r=await runner.observe(event('write',torch.eye(16)[i],i),should_respond=False)
                if first is None:first=r
                await backend.processed(r)
            self.assertEqual(len(backend.states),2)
            with self.assertRaises(ValueError):await backend.processed(first)
        finally:
            await runner.close();await backend.aclose()


class FastEventValidationTests(unittest.TestCase):
    def test_target_and_nonfinite_forbidden(self):
        valid={'op':'query','key':torch.eye(16)[0].tolist()}
        for bad in ({**valid,'target':2},{**valid,'key':[float('nan')]*16},{**valid,'key':[0]*16}):
            with self.assertRaises(ValueError):parse_fast_event(json.dumps(bad))


if __name__=='__main__':unittest.main()
