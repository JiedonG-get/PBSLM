"""Fixed-model context transport audit with active input/output interruptions."""
import asyncio
import json
from pathlib import Path

from runtime.continuous import ContinuousRunner
from runtime.pbslm_context_backend import PBSLMContextBackend
from runtime.model_session import model_fingerprint,state_hash
from evaluation.text_facts import TEMPLATES,ENTITIES
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


async def execute():
    backend=PBSLMContextBackend(ROOT/'experiments/text_runtime/models800/seed0.pt',
                                 max_new_tokens=6,history_tokens=3,prefix_tokens=3,step_delay=.01)
    params=model_fingerprint(backend.model);memory=state_hash(backend.model,memory=True)
    runner=ContinuousRunner(backend)
    try:
        for entity in range(3):
            revision=await runner.observe(TEMPLATES[0].format(entity=ENTITIES[entity]))
            await runner.wait_event(lambda e:e['kind']=='output_committed' and e['generation_revision']==revision,timeout=15)
        await runner.wait_event(lambda e:e['kind']=='generation_finished' and e['generation_revision']==revision,timeout=15)
    finally:
        try:await runner.close()
        finally:await asyncio.to_thread(backend.shutdown)
    if params!=model_fingerprint(backend.model) or memory!=state_hash(backend.model,memory=True):
        raise AssertionError('fixed model changed')
    for packed in backend.packing:
        actual=[step for step in backend.metrics if step['revision']==packed['revision'] and step['step']==0]
        if len(actual)!=1 or actual[0]['actual_compact_input_ids']!=packed['initial_compact_ids']:
            raise AssertionError('reported packed context differs from actual neural input')
    for packed in backend.packing[1:]:
        roles={span['role'] for span in packed['spans']}
        if not {'prior_input','committed_output','latest_input'}<=roles:
            raise AssertionError('continued request did not carry all context roles')
    if any(e['revision']!=e['generation_revision'] for e in runner.journal if e['kind']=='output_committed'):
        raise AssertionError('stale output')
    return {'scope':'context token transport, not semantic continuity or natural language quality',
            'checkpoint':backend.artifact,'model_and_memory_unchanged':True,'output':runner.output,
            'journal':runner.journal,'packing':backend.packing,'neural_steps':backend.metrics,
            'all_reported_inputs_match_model_step0':True,'continued_requests_include_three_roles':True,
            'no_stale_commits':True,'context_layout_not_trained':True}


def main():
    out=ROOT/'experiments/continuous_online/context_transport.json'
    if out.exists():raise FileExistsError(out)
    result=asyncio.run(execute());result['provenance']=capture(['runtime/event_context.py','runtime/pbslm_context_backend.py',
                         'runtime/pbslm_token_backend.py','runtime/continuous.py','evaluation/context_transport_demo.py'])
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('all_reported_inputs_match_model_step0','continued_requests_include_three_roles',
                                           'model_and_memory_unchanged','no_stale_commits')}))


if __name__=='__main__':main()
