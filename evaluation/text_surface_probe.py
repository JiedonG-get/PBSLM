"""Frozen persisted models on new text surfaces; representation coverage != accuracy."""
import argparse
import json
from pathlib import Path
import statistics

import torch
from evaluation.text_checkpoint import load
from evaluation.text_facts import ENTITIES, fill, frozen_hash
from evaluation.provenance import capture
from runtime.text_codec import CompactTextCodec, InputContractError

ROOT=Path(__file__).resolve().parents[1]
NEW_TEMPLATES=['{entity}的颜色？','颜色是什么，{entity}？','{entity}颜色是什么？',
               '请回答{entity}的颜色。','告诉我{entity}是什么颜色。',
               '我想知道，{entity}是什么颜色？','请说出{entity}是什么颜色的？',
               '我想知道{entity}的颜色是什么？','请告诉我{entity}的颜色。',
               '{entity}，是什么颜色？','关于{entity}，它的颜色是？',
               '麻烦查询一下{entity}现在的颜色。']


@torch.no_grad()
def evaluate(seed,args):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,task,metadata=load(ROOT/args.models/f'seed{seed}.pt',device)
    codec=CompactTextCodec(task.original_ids,task.length,ROOT/'tokenizer/bpe16k_tokenizer.json',task.tokenizer_hash)
    digest=frozen_hash(model); rows=[]; metrics={}
    for split in ('familiar','novel'):
        gen=torch.Generator().manual_seed(seed+19000+(split=='novel'))
        for episode in range(args.episodes):
            sx,qx,hx,y,meta=task.episode(gen,split)
            for template_index,template in enumerate(NEW_TEMPLATES):
                supported=[]
                for j,entity in enumerate(meta['entities']):
                    text=template.format(entity=ENTITIES[entity])
                    try:
                        encoding=codec.inspect(text)
                        supported.append((j,text,encoding))
                    except InputContractError as error:
                        rows.append({'split':split,'episode':episode,'template':template_index,'entity':entity,
                                     'text':text,'accepted':False,'reason':error.reason,'details':error.details})
                for mode in ('normal','zero','wrong'):
                    metric=metrics.setdefault(f'{split}/{template_index}/{mode}',
                                              {'attempted':0,'accepted':0,'correct':0,'written_matches':0})
                    metric['attempted']+=len(y); metric['accepted']+=len(supported)
                    if not supported:continue
                    fill(model,sx,y,mode)
                    inputs=torch.tensor([item[2]['compact_ids'] for item in supported],device=device)
                    out=model(inputs,1,sample_noise=False)
                    predicted=model.decode_nearest(out['z_a']).argmax(-1)[:,0]
                    targets=y[torch.tensor([item[0] for item in supported],device=device)]
                    written=(y.roll(1) if mode=='wrong' else y)[torch.tensor([item[0] for item in supported],device=device)]
                    metric['correct']+=int((predicted==targets).sum())
                    metric['written_matches']+=int((predicted==written).sum())
                    for prediction,target,(j,text,encoding) in zip(predicted.tolist(),targets.tolist(),supported):
                        rows.append({'split':split,'episode':episode,'template':template_index,
                                     'entity':meta['entities'][j],'text':text,'accepted':True,'mode':mode,
                                     'prediction':prediction,'target':target,'encoding':encoding})
    for item in metrics.values():
        item['coverage']=item['accepted']/item['attempted']
        item['accepted_accuracy']=item['correct']/item['accepted'] if item['accepted'] else None
        item['correct_over_all_attempts']=item['correct']/item['attempted']
    if frozen_hash(model)!=digest:raise AssertionError('evaluation altered model state')
    return {'seed':seed,'checkpoint_sha256':metadata['sha256'],'metrics':metrics,'traces':rows,
            'parameters_unchanged':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--episodes',type=int,default=16);p.add_argument('--models',default='experiments/text_runtime/models800')
    p.add_argument('--output',default='experiments/text_surfaces/frozen_probe.json');args=p.parse_args()
    if args.episodes<1:p.error('episodes must be positive')
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    report={'config':vars(args),'new_templates':NEW_TEMPLATES,
            'scope':'narrow manually authored surfaces, fixed compact vocab and frozen learned weights',
            'provenance':capture(['evaluation/text_surface_probe.py','runtime/text_codec.py','evaluation/text_checkpoint.py',
                                  'evaluation/text_facts.py','models/model.py','models/model_v2.py']), 'runs':[]}
    for seed in args.seeds:
        result=evaluate(seed,args);report['runs'].append(result)
        totals={split:{k:sum(v[k] for name,v in result['metrics'].items() if name.startswith(split+'/') and name.endswith('/normal'))
                       for k in ('attempted','accepted','correct')} for split in ('familiar','novel')}
        print(json.dumps({'seed':seed,'normal_totals':totals}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
