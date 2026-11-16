"""Actual repeated neural token steps interrupted by new input; no training."""
import asyncio
import json
from pathlib import Path

from runtime.continuous import ContinuousRunner
from runtime.pbslm_token_backend import PBSLMTokenBackend
from runtime.model_session import model_fingerprint,state_hash
from evaluation.provenance import capture
from evaluation.text_facts import ENTITIES,TEMPLATES

ROOT=Path(__file__).resolve().parents[1]


async def run():
    backend=PBSLMTokenBackend(ROOT/'experiments/text_runtime/models800/seed0.pt',max_new_tokens=8,step_delay=.01)
    initial=model_fingerprint(backend.model);memory=state_hash(backend.model,memory=True)
    runner=ContinuousRunner(backend)
    prefixes=[]
    try:
        for i in range(3):
            revision=await runner.observe(TEMPLATES[0].format(entity=ENTITIES[i]))
            await runner.wait_event(lambda e:e['kind']=='output_committed' and e['generation_revision']==revision,timeout=15)
            prefixes.append(runner.output)
        final=runner.revision
        await runner.wait_event(lambda e:e['kind']=='generation_finished' and e['generation_revision']==final,timeout=15)
    finally:
        try:await runner.close()
        finally:await asyncio.to_thread(backend.shutdown)
    if initial!=model_fingerprint(backend.model) or memory!=state_hash(backend.model,memory=True):
        raise AssertionError('fixed model or M changed')
    if not runner.output.startswith(prefixes[0]):raise AssertionError('committed prefix lost')
    if any(e['generation_revision']!=e['revision'] for e in runner.journal if e['kind']=='output_committed'):
        raise AssertionError('stale neural token committed')
    return {'scope':'real bounded neural rollout scheduling, NOT fluent language or semantic resume',
            'output':runner.output,'prefixes_at_interrupt':prefixes,'journal':runner.journal,
            'neural_steps':backend.metrics,'checkpoint':backend.artifact,
            'fixed_model_and_memory':True,'no_stale_token_commits':True,
            'rendered_prefix_not_model_conditioning':True,'step_delay_explicit_seconds':.01}


def main():
    out=ROOT/'experiments/continuous_online/neural_token_rollout.json'
    if out.exists():raise FileExistsError(out)
    result=asyncio.run(run());result['provenance']=capture(['runtime/continuous.py','runtime/pbslm_token_backend.py','evaluation/neural_token_online_demo.py'])
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'neural_steps':len(result['neural_steps']),'committed_units':len([e for e in result['journal'] if e['kind']=='output_committed']),
                      'no_stale_token_commits':True,'quality':'not evaluated; model not trained for fluent rollout'}))


if __name__=='__main__':main()
