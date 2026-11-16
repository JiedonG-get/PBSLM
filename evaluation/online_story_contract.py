"""Public online-story development fixtures and separate prediction scorer.

input_view strips all reference labels before invoking a future backend. This file
runs no model. Exact answer string scoring is a strict contract, not semantic judge.
Interrupt labels are authored policy references, not universal human preferences.
"""
import argparse
import json
from pathlib import Path

FIXTURE=Path(__file__).resolve().parent/'fixtures/online_stories_dev.jsonl'


def load_cases(path=FIXTURE):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data['format']!='pbslm.online_story_cases.v1':raise ValueError('fixture version mismatch')
    seen=set()
    for case in data['cases']:
        if not case['events']:raise ValueError('nonempty case required')
        for event in case['events']:
            key=(case['id'],event['id'])
            if key in seen or type(event['interrupt']) is not bool:raise ValueError('invalid fixture key/label')
            seen.add(key)
    return data


def input_view(case,through_event):
    if type(through_event) is not int or not 0<=through_event<len(case['events']):
        raise ValueError('event index outside case')
    return {'case_id':case['id'],'question':case['question'],
            'events':[{'id':e['id'],'text':e['text']} for e in case['events'][:through_event+1]]}


def score(data,predictions):
    if not isinstance(predictions,list):raise ValueError('prediction rows required')
    expected={(c['id'],e['id']):e for c in data['cases'] for e in c['events']}
    submitted={}
    for row in predictions:
        key=(row['case_id'],row['event_id'])
        if key not in expected or key in submitted:raise ValueError('unknown or duplicate prediction event')
        if not isinstance(row['answer'],str) or type(row['interrupt']) is not bool:
            raise ValueError('answer string and interrupt boolean required')
        submitted[key]=row
    exact=0;missed=0;false=0;rows=[]
    for key,event in expected.items():
        prediction=submitted.get(key)
        hit=prediction is not None and prediction['answer']==event['target_after']
        exact+=hit
        missed+=int(event['interrupt'] and (prediction is None or not prediction['interrupt']))
        false+=int(not event['interrupt'] and prediction is not None and prediction['interrupt'])
        rows.append({'case_id':key[0],'event_id':key[1],'submitted':prediction is not None,
                     'answer_exact':hit,'reference_interrupt':event['interrupt']})
    relevant=sum(e['interrupt'] for e in expected.values());irrelevant=len(expected)-relevant
    return {'events':len(expected),'submitted':len(submitted),'coverage':len(submitted)/len(expected),
            'answer_exact_all_events':exact/len(expected),
            'missed_required_interrupts':missed,'false_interrupts':false,
            'required_interrupt_events':relevant,'non_interrupt_events':irrelevant,
            'rows':rows,'limits':['exact answer labels, no paraphrase normalization',
                                  'missing prediction counts wrong/missed required interrupt',
                                  'does not measure actual scheduling delay or output progress',
                                  'public authored dev cases, no generalization claim']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predictions',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    result=score(load_cases(),json.loads(Path(args.predictions).read_text(encoding='utf-8')))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
