"""Separate surface calibration/test of frozen reader; no training or memory writes.

Select policy on calibration templates only, then evaluate on distinct test templates.
Sample precision constraints do not certify safety under template distribution shift.
"""
import argparse
import json
from pathlib import Path
import random

import torch
from runtime.fact_reader import load_reader,encode
from evaluation.text_facts import ENTITIES,COLORS
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]
CAL_POS=['确认记录：{e}的颜色是{c}。','刚刚记录了{e}呈{c}色。','当前信息是{e}为{c}色。']
CAL_NEG=['{e}不一定是{c}色。','我没有确认{e}是{c}色。','请问{e}是否为{c}色？',
         '据猜测，{e}可能呈{c}色。','过去{e}是{c}色，现在状态未知。']
TEST_POS=['已核实，{e}目前呈{c}色。','最新记录写着：{e}为{c}色。','现在可确定{e}的颜色为{c}。']
TEST_NEG=['还无法确定{e}是否是{c}色。','传闻{e}为{c}色，但尚未证实。',
          '别把{e}当成{c}色。','{e}将来或许会变成{c}色。','谁说{e}现在是{c}色？',
          '只有在测试中{e}才被设为{c}色。']


def examples(seed,n,positive,negative):
    rng=random.Random(seed);out=[]
    for i in range(n):
        valid=i%2==0;e=rng.randrange(16);c=rng.randrange(8)
        templates=positive if valid else negative;t=rng.randrange(len(templates))
        out.append({'text':templates[t].format(e=ENTITIES[e],c=COLORS[c]),
                    'valid':valid,'entity':e,'color':c,'template':t})
    return out


@torch.no_grad()
def score(model,tok,rows,device):
    results=[]
    for i in range(0,len(rows),64):
        batch=rows[i:i+64];ids,lengths=encode([x['text'] for x in batch],tok,device)
        v,e,c=model(ids,lengths)
        if any(not torch.isfinite(x).all() for x in (v,e,c)):raise FloatingPointError('reader logits')
        vp=v.softmax(-1)[:,1];ep=e.softmax(-1).topk(2,dim=-1);cp=c.softmax(-1).topk(2,dim=-1)
        for b,row in enumerate(batch):
            results.append({**row,'valid_score':float(vp[b]),'pred_entity':int(ep.indices[b,0]),
                            'pred_color':int(cp.indices[b,0]),
                            'entity_margin':float(ep.values[b,0]-ep.values[b,1]),
                            'color_margin':float(cp.values[b,0]-cp.values[b,1])})
    return results


def metric(rows,threshold,margin):
    accepted=[r for r in rows if r['valid_score']>=threshold and r['entity_margin']>=margin and r['color_margin']>=margin]
    positive=sum(r['valid'] for r in rows);negative=len(rows)-positive
    exact=lambda r:r['valid'] and r['entity']==r['pred_entity'] and r['color']==r['pred_color']
    good=sum(exact(r) for r in accepted)
    return {'threshold':threshold,'margin':margin,'n':len(rows),'accepted':len(accepted),
            'correct_accepted':good,'wrong_accepted':len(accepted)-good,
            'precision':good/len(accepted) if accepted else None,
            'positive_coverage':sum(r['valid'] for r in accepted)/positive,
            'negative_false_accept_rate':sum(not r['valid'] for r in accepted)/negative}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--examples',type=int,default=512);p.add_argument('--minimum-accepted',type=int,default=20)
    p.add_argument('--output',default='experiments/fact_reader/calibration_shift.json');args=p.parse_args()
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    if args.examples<40 or args.examples%2:p.error('even examples>=40 required')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    report={'config':vars(args),'policy_rule':'max accepted on calibration at empirical precision>=.99 and accepted>=20; otherwise reject all',
            'calibration_templates':{'positive':CAL_POS,'negative':CAL_NEG},
            'test_templates':{'positive':TEST_POS,'negative':TEST_NEG},
            'provenance':capture(['evaluation/calibrate_fact_reader.py','runtime/fact_reader.py']), 'runs':[]}
    for seed in args.seeds:
        model,tok,metadata=load_reader(ROOT/f'experiments/fact_reader/three_seed600/seed{seed}.pt',device)
        cal=score(model,tok,examples(seed+31000,args.examples,CAL_POS,CAL_NEG),device)
        test=score(model,tok,examples(seed+32000,args.examples,TEST_POS,TEST_NEG),device)
        grid=[metric(cal,t,m) for t in (.5,.7,.9,.95,.99,.995,.999,.9999) for m in (0.,.1,.2,.4,.6)]
        eligible=[x for x in grid if x['accepted']>=args.minimum_accepted and x['precision']>=.99]
        selected=max(eligible,key=lambda x:(x['accepted'],x['threshold'],x['margin'])) if eligible else None
        held=metric(test,selected['threshold'],selected['margin']) if selected else {'accepted':0,'precision':None,'positive_coverage':0,'negative_false_accept_rate':0,'policy':'reject_all'}
        row={'seed':seed,'checkpoint':metadata,'calibration_grid':grid,'selected':selected,
             'test_selected_policy':held,'test_original_policy':metric(test,.9,.2),
             'calibration_traces':cal,'test_traces':test}
        report['runs'].append(row)
        print(json.dumps({k:row[k] for k in ('seed','selected','test_selected_policy','test_original_policy')}),flush=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
