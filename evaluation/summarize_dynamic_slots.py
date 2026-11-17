"""Check stored counters and group-wise means for explicit dynamic-slot reference."""
import json
from pathlib import Path
import statistics


def main():
    source=Path('experiments/recurrent_state/content_fast600/dynamic_slots.json');out=source.with_name('dynamic_slots_summary.json')
    if out.exists():raise FileExistsError(out)
    data=json.loads(source.read_text(encoding='utf-8'));summary={}
    for run in data['runs']:
        for result in run['evaluations'].values():
            for method in ('learned','key_value_bank_reference'):
                for counts in result[method].values():
                    if abs(counts['correct']/counts['n']-counts['accuracy'])>1e-12:raise AssertionError('counter mismatch')
    for condition in data['runs'][0]['evaluations']:
        summary[condition]={group:{'mean':statistics.mean(r['evaluations'][condition]['learned'][group]['accuracy'] for r in data['runs']),
                                  'per_seed':[r['evaluations'][condition]['learned'][group]['accuracy'] for r in data['runs']]}
                            for group in ('known','unknown')}
    out.write_text(json.dumps({'source':str(source),'summary':summary,'note':'counter arithmetic only; no per-event predictions saved'},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
