"""Recount all-hot frozen gate probes and stratify cold entities by route collision."""
import json
from pathlib import Path
import statistics


def main():
    source=Path('experiments/recurrent_state/all_hot2048.json');out=source.with_name('all_hot2048_summary.json')
    if out.exists():raise FileExistsError(out)
    data=json.loads(source.read_text(encoding='utf-8'));summary={};count_cases=0
    for run in data['runs']:
        for case in run['cases']:
            count_cases+=1;hot=case['hot_entity']
            for mode,rows in case['records'].items():
                for row in rows:
                    counts=[sum(p[e]==t[e] for p,t in zip(row['predicted_ids'],row['target_ids'])) for e in range(8)]
                    if counts!=row['per_entity_correct'] or counts[hot]!=row['hot_correct'] or sum(counts[e] for e in range(8) if e!=hot)!=row['cold_correct']:
                        raise AssertionError('prediction count mismatch')
            if case['records']['soft'][0]['predicted_ids']!=case['records']['top1_mass'][0]['predicted_ids']:
                raise AssertionError('priming differs between arms')
    for mode in ('soft','top1_mass'):
        cases=[case for run in data['runs'] for case in run['cases']]
        total={'cases':len(cases),'initial_cold_mean':statistics.mean(c['records'][mode][0]['cold_correct']/c['records'][mode][0]['cold_n'] for c in cases),
               'final_cold_mean':statistics.mean(c['records'][mode][-1]['cold_correct']/c['records'][mode][-1]['cold_n'] for c in cases),
               'final_hot_mean':statistics.mean(c['records'][mode][-1]['hot_correct']/c['records'][mode][-1]['per_entity_n'] for c in cases)}
        for group in ('colliding','noncolliding'):
            correct=initial=n=0
            for c in cases:
                ids=[e for e in range(8) if e!=c['hot_entity'] and ((e in c['colliding_cold_entities'])==(group=='colliding'))]
                for e in ids:
                    correct+=c['records'][mode][-1]['per_entity_correct'][e]
                    initial+=c['records'][mode][0]['per_entity_correct'][e]
                    n+=c['records'][mode][-1]['per_entity_n']
            total[group]={'n':n,'initial_accuracy':initial/n if n else None,'final_accuracy':correct/n if n else None}
        summary[mode]=total
    report={'source':str(source),'total_hot_cases':count_cases,'arms':summary,
            'note':'24 seed/hot cases, collision probes mostly one trained checkpoint; not iid significance'}
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
