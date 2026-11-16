"""Trace-based descriptive summaries; no neural run and no overwrite."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='experiments/consolidation/shared_soft200.json')
    parser.add_argument('--output',default='experiments/consolidation/shared_soft200_summary.json');args=parser.parse_args()
    target=Path(args.output)
    if target.exists():raise FileExistsError(target)
    data=json.loads(Path(args.input).read_text(encoding='utf-8'))
    for run in data['runs']:
        if set(run['old_ids'])!=set(run['new_ids']):raise AssertionError('answer sets not shared')
        for evaluation in [run['before']]+[arm['evaluation'] for arm in run['arms'].values()]:
            for name,item in evaluation.items():
                actual=sum(a==b for a,b in zip(item['predicted_ids'],item['target_ids']))/len(item['target_ids'])
                if abs(actual-item['accuracy'])>1e-6:raise AssertionError('trace/metric mismatch')
    summary={'source':args.input,'seeds':[r['seed'] for r in data['runs']],
             'before':{key:statistics.mean(r['before'][key]['accuracy'] for r in data['runs']) for key in data['runs'][0]['before']},
             'arms':{arm:{key:{'mean':statistics.mean(r['arms'][arm]['evaluation'][key]['accuracy'] for r in data['runs']),
                               'per_seed':[r['arms'][arm]['evaluation'][key]['accuracy'] for r in data['runs']]}
                          for key in data['runs'][0]['before']} for arm in data['runs'][0]['arms']},
             'teacher':{mode:{'top1_mean':statistics.mean(r['teacher'][mode]['top1_truth'] for r in data['runs']),
                              'entropy_mean':statistics.mean(r['teacher'][mode]['mean_entropy'] for r in data['runs'])}
                         for mode in ('normal','zero','wrong')},
             'note':'three-seed descriptive mean, eight correlated queries per metric; no significance claim'}
    target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
