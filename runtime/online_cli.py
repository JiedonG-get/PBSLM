"""Terminal experience of continuous input/output. No training.

python -m runtime.online_cli --backend reference
python -m runtime.online_cli --backend pbslm-query --checkpoint <local text checkpoint>
The reference backend is symbolic/rule-rendered; PBSLM is restricted color queries.
Neither is a general chatbot or autonomous speech-policy demonstration.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

from .continuous import ContinuousRunner
from .continuous_state import ContinuousStateBackend,ExactReferenceExtractor,LearnedReaderExtractor
from .pbslm_backend import PBSLMQueryBackend


async def serve(args):
    if args.backend=='reference':backend=ContinuousStateBackend(ExactReferenceExtractor(),unit_delay=args.delay)
    elif args.backend=='reader':backend=ContinuousStateBackend(LearnedReaderExtractor(args.reader),unit_delay=args.delay)
    else:backend=PBSLMQueryBackend(args.checkpoint,unit_delay=args.delay)
    runner=ContinuousRunner(backend)
    print('PBSLM continuous Online prototype: fixed weights, NOT a general chatbot.',flush=True)
    print('Commands: /listen TEXT | /interrupt | /resume | /wait | /quit. Ordinary text requests output.',flush=True)
    print('Reference/reader state updates: 甲号设备的颜色是红。 ; query: 甲号设备的颜色是什么？',flush=True)
    if args.backend=='reader':print('WARNING: reader may accept false/negated claims; this is unverified ephemeral state, NOT numerical M.',flush=True)
    if args.backend=='pbslm-query':print('WARNING: one-token color-query model; no persistent understanding or fluent dialogue.',flush=True)

    async def display():
        cursor=0
        while True:
            cursor,events=await runner.read_events(cursor,timeout=None)
            for event in events:
                if event['kind']=='output_committed':
                    print(f"OUT r{event['generation_revision']}: {event['unit']}",flush=True)
                elif event['kind'] in ('generation_failed','response_deferred','shutdown_timeout'):
                    print('STATUS '+json.dumps(event,ensure_ascii=False),flush=True)
            if runner.closed and not events:return
            if events and events[-1]['kind']=='closed':return
    printer=asyncio.create_task(display())
    try:
        while True:
            line=await asyncio.to_thread(sys.stdin.readline)
            if not line:break
            line=line.rstrip('\r\n')
            if not line:continue
            if line=='/quit':break
            try:
                if line=='/interrupt':await runner.interrupt('terminal_interrupt')
                elif line=='/resume':await runner.resume()
                elif line=='/wait':
                    # Explicit barrier for piped demos; ordinary interactive input never waits.
                    input_revision=runner.inputs[-1][0] if runner.inputs else None
                    if input_revision is not None and hasattr(backend,'processed'):
                        await asyncio.wait_for(backend.processed(input_revision),timeout=10)
                    generation=runner.revision
                    if runner._active is not None or runner._pending_revision==generation:
                        await runner.wait_event(lambda e:e['kind'] in ('generation_finished','generation_failed')
                                                and e['generation_revision']==generation,timeout=10)
                elif line.startswith('/listen '):await runner.observe(line[8:],should_respond=False)
                elif line.startswith('/'):
                    print('Unknown command',flush=True)
                else:await runner.observe(line)
            except (ValueError,RuntimeError,asyncio.TimeoutError,asyncio.QueueFull) as exc:
                print('ERROR '+type(exc).__name__+': '+str(exc),flush=True)
    finally:
        try:await runner.close()
        finally:
            try:
                if hasattr(backend,'aclose'):await backend.aclose()
                else:await asyncio.to_thread(backend.shutdown)
            finally:
                try:
                    await asyncio.wait_for(asyncio.shield(printer),timeout=2)
                except (asyncio.TimeoutError,asyncio.CancelledError):
                    printer.cancel()
                await asyncio.gather(printer,return_exceptions=True)
        if args.trace:
            path=Path(args.trace)
            if path.exists():raise FileExistsError(path)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'backend':args.backend,'runtime':runner.journal,
                           'backend_trace':getattr(backend,'journal',getattr(backend,'metrics',[])),
                           'no_training':True},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['reference','reader','pbslm-query'],default='reference')
    p.add_argument('--reader',default='experiments/fact_reader/three_seed600/seed0.pt')
    p.add_argument('--checkpoint',default='experiments/text_runtime/models800/seed0.pt')
    p.add_argument('--delay',type=float,default=.15);p.add_argument('--trace',default='')
    args=p.parse_args()
    if args.delay<0:p.error('delay must be nonnegative')
    if args.trace and Path(args.trace).exists():p.error('trace file already exists')
    asyncio.run(serve(args))


if __name__=='__main__':main()
