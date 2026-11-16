"""Paired synthetic sentence reader train/eval; no PBSLM parameter changes.

Entity/color identity supervised. Test statements have held templates but familiar
inventories. Negative held forms test abstention risk; not a general NLP benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from runtime.fact_reader import CandidateReader,encode,infer,tokenizer
from evaluation.text_facts import ENTITIES,COLORS
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]
POS_TRAIN=['{e}的颜色是{c}。','现在{e}是{c}色。','记录：{e}，颜色为{c}。','把{e}的颜色更新为{c}。']
POS_HELD=['最新消息：{e}目前的颜色是{c}。','{e}现在呈{c}色。',
          '请记住，{e}的颜色为{c}。','设备记录显示{e}是{c}色。']
NEG_TRAIN=['{e}的颜色是什么？','{e}不是{c}色。','我猜{e}可能是{c}色。',
           '{e}的颜色未知。','{e}是{c}色，{other}是{c2}色。']
NEG_HELD=['你知道{e}是什么颜色吗？','不能确认{e}是不是{c}色。',
          '有人说{e}也许是{c}色。','{e}或者{other}是{c}色。',
          '{e}以前是{c}色，现在不知道。','如果{e}是{c}色就好了。']


def example(rng,positive=True,held=False):
    e=rng.randrange(16);c=rng.randrange(8);other=(e+rng.randrange(1,16))%16;c2=(c+rng.randrange(1,8))%8
    templates=(POS_HELD if held else POS_TRAIN) if positive else (NEG_HELD if held else NEG_TRAIN)
    index=rng.randrange(len(templates))
    return {'text':templates[index].format(e=ENTITIES[e],c=COLORS[c],other=ENTITIES[other],c2=COLORS[c2]),
            'valid':int(positive),'entity':e,'color':c,'template':index}


def metrics(rows):
    positive=[x for x in rows if x['truth']['valid']];negative=[x for x in rows if not x['truth']['valid']]
    accepted=[x for x in rows if x['prediction']['accepted']]
    correct=lambda x: bool(x['truth']['valid'] and x['truth']['entity']==x['prediction']['entity'] and x['truth']['color']==x['prediction']['color'])
    return {'n':len(rows),'positive_n':len(positive),'negative_n':len(negative),'accepted':len(accepted),
            'positive_coverage':sum(x['prediction']['accepted'] for x in positive)/len(positive),
            'positive_exact_accuracy_unconditional':sum(correct(x) for x in positive)/len(positive),
            'accepted_exact_precision':sum(correct(x) for x in accepted)/len(accepted) if accepted else None,
            'negative_false_accept_rate':sum(x['prediction']['accepted'] for x in negative)/len(negative),
            'accepted_wrong_candidates':sum(not correct(x) for x in accepted)}


def run(seed,args):
    torch.manual_seed(seed);device='cuda' if torch.cuda.is_available() else 'cpu'
    tok,tokhash=tokenizer();model=CandidateReader(tok.get_vocab_size()).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.002)
    rng=random.Random(seed+21000);curve=[];start=time.perf_counter()
    for step in range(args.steps):
        batch=[example(rng,positive=i%2==0) for i in range(32)]
        ids,lengths=encode([x['text'] for x in batch],tok,device)
        valid,entity,color=model(ids,lengths)
        truth=torch.tensor([x['valid'] for x in batch],device=device)
        positive=truth.bool()
        objective=F.cross_entropy(valid,truth)+F.cross_entropy(entity[positive],torch.tensor([x['entity'] for x in batch],device=device)[positive])+F.cross_entropy(color[positive],torch.tensor([x['color'] for x in batch],device=device)[positive])
        opt.zero_grad(set_to_none=True);objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
        if (step+1)%100==0:curve.append({'step':step+1,'loss':float(objective.detach())})
    model.eval();results={}
    for split,held in [('seen_templates',False),('held_templates',True)]:
        gen=random.Random(seed+22000+(100 if held else 0))
        examples=[example(gen,i%2==0,held) for i in range(args.test_examples)]
        predictions=[]
        for i in range(0,len(examples),64):
            predictions.extend(infer(model,[x['text'] for x in examples[i:i+64]],tok,device))
        rows=[{'truth':x,'prediction':y} for x,y in zip(examples,predictions)]
        results[split]={'metrics':metrics(rows),'traces':rows}
    path=ROOT/args.output_dir/f'seed{seed}.pt'
    payload={'format':'pbslm.fact_reader.v1','state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
             'config':{'vocab':tok.get_vocab_size(),'embedding':32,'hidden':48},'tokenizer_sha256':tokhash,
             'entities':ENTITIES,'colors':COLORS,'seed':seed,'steps':args.steps,'threshold':.9,'margin':.2}
    torch.save(payload,path)
    return {'seed':seed,'seconds':time.perf_counter()-start,'curve':curve,'results':results,
            'checkpoint':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size},
            'parameters':sum(p.numel() for p in model.parameters())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--steps',type=int,default=600)
    p.add_argument('--test-examples',type=int,default=512);p.add_argument('--output-dir',default='experiments/fact_reader/pilot600')
    args=p.parse_args();out=ROOT/args.output_dir
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    report={'config':vars(args),'positive_train':POS_TRAIN,'positive_held':POS_HELD,
            'negative_train':NEG_TRAIN,'negative_held':NEG_HELD,
            'scope':'separate supervised bounded relation reader, familiar entity/color inventories',
            'provenance':capture(['evaluation/train_fact_reader.py','runtime/fact_reader.py']), 'runs':[]}
    for seed in args.seeds:
        result=run(seed,args);report['runs'].append(result)
        print(json.dumps({'seed':seed,'seconds':result['seconds'],'metrics':{k:v['metrics'] for k,v in result['results'].items()}}),flush=True)
    (out/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
