"""Continuous silent updates + incoming correction during speech.

Reference exact parser and separate fixed learned GRU compared. State/rendering are
symbolic, not PBSLM's semantic understanding. No M writes, no training/weight change.
"""
import asyncio
import hashlib
import json
from pathlib import Path

from runtime.continuous import ContinuousRunner
from runtime.continuous_state import ContinuousStateBackend,ExactReferenceExtractor,LearnedReaderExtractor
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


def reader_hash(extractor):
    if not hasattr(extractor,'model'):return None
    h=hashlib.sha256()
    for name,value in extractor.model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


async def scenario(extractor):
    backend=ContinuousStateBackend(extractor,unit_delay=.05)
    runner=ContinuousRunner(backend)
    frozen=reader_hash(extractor)
    try:
        first=await runner.observe('甲号设备的颜色是红。',should_respond=False)
        initial=await backend.processed(first)
        if 0 not in initial:raise AssertionError('initial fact not extracted')
        if runner.output:raise AssertionError('passive fact unexpectedly spoke')
        query=await runner.observe('甲号设备的颜色是什么？')
        await runner.wait_event(lambda e:e['kind']=='output_committed' and e['generation_revision']==query,timeout=5)
        prefix=runner.output
        correction=await runner.observe('把甲号设备的颜色更新为绿。')
        updated=await backend.processed(correction)
        await runner.wait_event(lambda e:e['kind']=='generation_finished' and e['generation_revision']==correction,timeout=5)
        if updated[0]['color']!=2:raise AssertionError('latest correction not applied')
        if '甲号设备的颜色记录为红。' in runner.output:raise AssertionError('old planned assertion committed')
        if '甲号设备的颜色记录为绿。' not in runner.output:raise AssertionError('new assertion missing')
        if not runner.output.startswith(prefix):raise AssertionError('committed prefix lost')
        # Even with no speaking, another entity's observation updates state.
        silent=await runner.observe('乙号设备的颜色是蓝。',should_respond=False)
        final_state=await backend.processed(silent)
    finally:
        try:await runner.close()
        finally:await backend.aclose()
    if reader_hash(extractor)!=frozen:raise AssertionError('reader weights changed')
    return {'extractor':type(extractor).__name__,'output':runner.output,
            'input_journal':backend.journal,'runtime_journal':runner.journal,
            'state':{str(k):v for k,v in final_state.items()},'silent_inputs_processed':True,
            'stale_red_assertion_suppressed':True,'updated_green_assertion_committed':True,
            'committed_prefix_preserved':True,'reader_weights_unchanged':True,
            'semantics_limit':'belief latest extracted assertion; rule renderer, not PBSLM response generation',
            'source_truth_not_verified':True}


def main():
    out=ROOT/'experiments/continuous_online/semantic_state_scenario.json'
    if out.exists():raise FileExistsError(out)
    async def execute():
        return {'reference':await scenario(ExactReferenceExtractor()),
                'learned_reader':await scenario(LearnedReaderExtractor(ROOT/'experiments/fact_reader/three_seed600/seed0.pt'))}
    report=asyncio.run(execute())
    report['scope']='fixed-weight continuous ingestion and explicit belief state with scripted response policy'
    report['provenance']=capture(['runtime/continuous.py','runtime/continuous_state.py','runtime/fact_reader.py',
                                  'evaluation/continuous_semantic_scenario.py'])
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({key:{field:value[field] for field in ('silent_inputs_processed','stale_red_assertion_suppressed',
                                      'updated_green_assertion_committed','reader_weights_unchanged')}
                      for key,value in report.items() if key in ('reference','learned_reader')}))


if __name__=='__main__':main()
