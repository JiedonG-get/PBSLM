"""Actual learned statement reader -> external confirm -> bounded PBSLM memory demo.

Fixed manually inspected seen-template examples. Success not held-generalization.
No weight training or update in this demo; saved local models required.
"""
import json
from pathlib import Path

from evaluation.text_checkpoint import load
from runtime.fact_reader import load_reader
from runtime.controlled_text import ControlledTextSession
from runtime.statement_candidates import StatementCandidates
from runtime.model_session import state_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'experiments/fact_reader/statement_demo.json'
    if out.exists():raise FileExistsError(out)
    model,task,metadata=load(ROOT/'experiments/text_runtime/models800/seed0.pt')
    reader,tok,reader_meta=load_reader(ROOT/'experiments/fact_reader/three_seed600/seed0.pt')
    session=ControlledTextSession(model,task,capacity=4)
    pipeline=StatementCandidates(session,reader,tok,threshold=reader_meta['threshold'],margin=reader_meta['margin'])
    immutable=state_hash(model);memory=state_hash(model,memory=True)
    trace=[]
    clock=0
    statements=['甲号设备的颜色是红。','乙号设备的颜色是蓝。',
                '甲号设备不是绿色。','把甲号设备的颜色更新为绿。']
    for text in statements:
        clock+=1
        item=pipeline.propose(text,event_time=clock)
        if state_hash(model,memory=True)!=memory:raise AssertionError('proposal wrote M')
        record={'candidate':item}
        # Human/operator decision scripted for this explicit demonstration, NOT model trust.
        if '不是' in text:
            pipeline.reject(item['candidate_id'],'explicit operator rejection of negated claim')
            record['decision']='external_reject'
        elif item['accepted']:
            clock+=1;record['receipt']=pipeline.confirm(item['candidate_id'],event_time=clock)
            record['decision']='external_confirm'
        else:record['decision']='reader_abstained'
        memory=state_hash(model,memory=True);trace.append(record)
    answers=[]
    for question in ['请回答甲号设备是什么颜色。','请回答乙号设备是什么颜色。']:
        clock+=1;answers.append(session.ask(question,event_time=clock))
    if immutable!=state_hash(model):raise AssertionError('parameter state changed')
    report={'scope':'scripted bounded relation candidate demo, explicit operator decisions, not autonomous fact verification',
            'checkpoint':metadata,'reader':reader_meta,'trace':trace,'answers':answers,
            'parameters_unchanged':True,'memory_slots':list(session.inner.controller.slots),
            'provenance':capture(['evaluation/statement_reader_demo.py','runtime/statement_candidates.py',
                                  'runtime/fact_reader.py','runtime/controlled_text.py'])}
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'answers':answers,'slots':report['memory_slots']},ensure_ascii=False))


if __name__=='__main__':main()
