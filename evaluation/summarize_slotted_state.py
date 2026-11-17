"""Aggregate paired symbolic-state outcomes; no models or training."""
import json
from pathlib import Path
import statistics


def main():
    source=Path('experiments/recurrent_state/slots600/results.json')
    target=Path('experiments/recurrent_state/slots600/summary.json')
    if target.exists():raise FileExistsError(target)
    data=json.loads(source.read_text(encoding='utf-8'));summary={}
    for arm in ('global','learned','explicit'):
        rows=[r['arms'][arm] for r in data['runs']]
        summary[arm]={'parameters':[r['parameters'] for r in rows],
                      'seconds':[r['seconds_training_and_eval'] for r in rows],
                      'accuracy':{key:{'mean':statistics.mean(r['evaluation'][key]['accuracy'] for r in rows),
                                       'per_seed':[r['evaluation'][key]['accuracy'] for r in rows]}
                                  for key in rows[0]['evaluation']},
                      'age_384':{},'top_slot_distinct':[r['routing']['distinct_top_slots'] if r['routing'] else None for r in rows]}
        for key in rows[0]['evaluation']:
            for row in rows:
                metric=row['evaluation'][key]
                if abs(metric['correct']/metric['n']-metric['accuracy'])>1e-12:raise AssertionError('count mismatch')
                if sum(v['n'] for v in metric['by_write_age'].values())!=metric['n']:raise AssertionError('age count mismatch')
        for age in ('0','1-3','4-15','16+'):
            correct=sum(r['evaluation']['384/state']['by_write_age'][age]['correct'] for r in rows)
            n=sum(r['evaluation']['384/state']['by_write_age'][age]['n'] for r in rows)
            summary[arm]['age_384'][age]={'correct':correct,'n':n,'accuracy':correct/n}
    target.write_text(json.dumps({'source':str(source),'arms':summary,'note':'three-seed descriptive means; age buckets pooled; no eventwise raw predictions'},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
