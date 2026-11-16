"""Derive event counts and restore write costs from existing traces; no model runs."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='experiments/text_runtime/stream96.json')
    p.add_argument('--output',default='experiments/text_runtime/accounting_audit.json')
    a=p.parse_args(); target=Path(a.output)
    if target.exists(): raise FileExistsError(target)
    data=json.loads(Path(a.input).read_text(encoding='utf-8')); rows=[]
    for run in data['runs']:
        for arm,result in run['arms'].items():
            events=result['events']
            for event in events:
                hits=sum(item['answer_a']==item['expected_latest_truth'] for item in event['probes'])
                if abs(hits/len(event['probes'])-event['accuracy'])>1e-12:
                    raise AssertionError('probe aggregate mismatch')
            restore_index=len(events)//2
            slots=events[restore_index]['retained_slots']
            restore_writes=len(slots) if result['capacity'] is not None else 0
            rows.append({'seed':run['seed'],'arm':arm,'event_count':len(events),
                         'correction_events':sum(e['correction'] for e in events),
                         'feedback_application_writes':result['numerical_write_calls'],
                         'restore_rebuild_writes_derived':restore_writes,
                         'total_numerical_writes_derived':result['numerical_write_calls']+restore_writes})
    report={'source':a.input,'model_executed':False,'raw_json_modified':False,
            'assumption':'single restore at len(events)//2, same source-code path; append restore copies tensors',
            'rows':rows,'mean_total_writes':{arm:statistics.mean(r['total_numerical_writes_derived'] for r in rows if r['arm']==arm)
                                            for arm in sorted({r['arm'] for r in rows})}}
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
