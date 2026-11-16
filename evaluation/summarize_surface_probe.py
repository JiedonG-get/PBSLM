"""Coverage-aware summary of immutable new-surface predictions; no model run."""
import argparse
import collections
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='experiments/text_surfaces/frozen_probe.json')
    p.add_argument('--output',default='experiments/text_surfaces/summary.json');args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    data=json.loads(Path(args.input).read_text(encoding='utf-8')); groups={}; rejections=collections.Counter()
    for run in data['runs']:
        for name,metric in run['metrics'].items():
            split,t,mode=name.split('/')
            item=groups.setdefault(f'{split}/{mode}',{'attempted':0,'accepted':0,'correct':0})
            for key in item:item[key]+=metric[key]
        for row in run['traces']:
            if not row['accepted']:rejections[row['reason']]+=1
    for item in groups.values():
        item['coverage']=item['accepted']/item['attempted']
        item['accepted_accuracy']=item['correct']/item['accepted'] if item['accepted'] else None
        item['correct_over_all_attempts']=item['correct']/item['attempted']
    per_template={}
    for i,text in enumerate(data['new_templates']):
        per_template[str(i)]={'text':text,'normal':{}}
        for split in ('familiar','novel'):
            records=[r['metrics'][f'{split}/{i}/normal'] for r in data['runs']]
            n=sum(x['attempted'] for x in records);a=sum(x['accepted'] for x in records);c=sum(x['correct'] for x in records)
            per_template[str(i)]['normal'][split]={'attempted':n,'accepted':a,'correct':c,
                    'coverage':a/n,'accepted_accuracy':c/a if a else None,'correct_over_all_attempts':c/n}
    report={'source':args.input,'groups':groups,'rejection_counts_per_question_not_mode':dict(rejections),
            'per_template':per_template,'note':'micro-averaged correlated template queries; no significance claim'}
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'groups':groups,'rejections':dict(rejections)},indent=2))


if __name__=='__main__':main()
