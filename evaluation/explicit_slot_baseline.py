"""Latest-feedback dictionary baseline on immutable real text Runtime traces.

Same caller-supplied entity slot and arrived feedback as matrix arms. Does NOT parse
language; this is a matching strong interface assumption, not an NLP baseline.
No model invocation/training; evaluator truth is used only to score probes.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import statistics
import time


def baseline(events,capacity):
    ledger=OrderedDict();trace=[]
    for event in events:
        key=event['entity']
        # Only the arrived feedback enters storage; hidden truth never enters ledger.
        ledger.pop(key,None);ledger[key]=event['feedback']
        if capacity is not None:
            while len(ledger)>capacity:ledger.popitem(last=False)
        predictions=[]
        for probe in event['probes']:
            answer=ledger.get(probe['entity'])
            predictions.append({'entity':probe['entity'],'target':probe['expected_latest_truth'],
                                 'answer':answer,'abstained':answer is None,
                                 'correct':answer==probe['expected_latest_truth']})
        accepted=[x for x in predictions if not x['abstained']]
        correct=sum(x['correct'] for x in predictions)
        trace.append({'event':event['event'],'feedback':event['feedback'],'entity':key,
                      'correction':event['correction'],'probes':predictions,
                      'n':len(predictions),'accepted':len(accepted),'correct':correct,
                      'accuracy_all':correct/len(predictions),'retained_slots':list(ledger)})
    n=sum(x['n'] for x in trace);a=sum(x['accepted'] for x in trace);c=sum(x['correct'] for x in trace)
    correction=[x for x,e in zip(trace,events) if e['correction']]
    return {'capacity':capacity,'events':trace,
            'mean_event_accuracy':statistics.mean(x['accuracy_all'] for x in trace),
            'micro_accuracy_all':c/n,'coverage':a/n,'accepted_precision':c/a if a else None,
            'final_accuracy':trace[-1]['accuracy_all'],
            'corrective_feedback_post_accuracy':sum(next(p['correct'] for p in x['probes'] if p['entity']==x['entity']) for x in correction)/len(correction),
            'dictionary_mutations':len(events),'max_retained_items':max(len(x['retained_slots']) for x in trace),
            'scope':'direct explicit slot lookup; no learned extraction or surface understanding'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='experiments/text_runtime/stream96.json')
    p.add_argument('--output',default='experiments/retrieval_baseline/explicit_slot.json');args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    source=json.loads(Path(args.input).read_text(encoding='utf-8'))
    report={'source':args.input,'model_executed':False,'training_executed':False,
            'assumptions':['same caller-provided entity slot','same arrived feedback, including wrong feedback',
                           'same update-recency eviction and capacity','abstention when slot absent counts wrong in all-probe accuracy',
                           'not equal memory bytes/FLOPs; Python overhead not GPU latency benchmark'], 'runs':[]}
    for run in source['runs']:
        start=time.perf_counter();arms={}
        for label,capacity,matrix in [('latest_all',None,'append'),('latest4',4,'bounded4'),('latest8',8,'bounded8')]:
            original=run['arms'][matrix]
            result=baseline(original['events'],capacity)
            result['matrix_comparator']={'arm':matrix,'mean_event_accuracy':original['average_accuracy'],
                                         'final_accuracy':original['final_accuracy'],
                                         'feedback_application_writes':original.get('feedback_application_write_calls',original['numerical_write_calls'])}
            if capacity is not None:
                for event,prior in zip(result['events'],original['events']):
                    expected=[f'entity:{key}/color' for key in event['retained_slots']]
                    if expected!=prior['retained_slots']:raise AssertionError('baseline and matrix eviction differ')
            arms[label]=result
        report['runs'].append({'seed':run['seed'],'arms':arms,'replay_seconds_not_model_latency':time.perf_counter()-start})
    report['means']={arm:{key:statistics.mean(r['arms'][arm][key] for r in report['runs'])
                          for key in ('mean_event_accuracy','micro_accuracy_all','coverage','accepted_precision',
                                      'final_accuracy','corrective_feedback_post_accuracy')}
                     for arm in ('latest_all','latest4','latest8')}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report['means'],indent=2))


if __name__=='__main__':main()
