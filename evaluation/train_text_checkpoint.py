"""Produce persisted key-aligned text models; inference artifacts stay local/ignored."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from evaluation.provenance import capture
from evaluation.text_facts import run
from evaluation.text_checkpoint import save, load

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--output-dir',default='experiments/text_runtime/models800')
    args=p.parse_args(); out=ROOT/args.output_dir
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    config=SimpleNamespace(episodes=800,eval_episodes=8,key_align_weight=.2)
    provenance=capture(['evaluation/train_text_checkpoint.py','evaluation/text_facts.py',
                        'evaluation/text_checkpoint.py','models/model.py','models/model_v2.py'])
    manifest={'training_config':vars(config),'provenance':provenance,'runs':[]}
    for seed in args.seeds:
        model,task,result=run(seed,config,return_model=True)
        artifact=save(out/f'seed{seed}.pt',model,task,{'seed':seed,**vars(config)},provenance)
        loaded,loaded_task,metadata=load(out/f'seed{seed}.pt',device='cpu')
        artifact['base_strict_load_ok']=True
        artifact['loaded_vocab']=loaded_task.vocab
        manifest['runs'].append({'seed':seed,'training_seconds':result['training_seconds'],
                                 'curve':result['curve'],'artifact':artifact,
                                 'evaluation':result['evaluation']['metrics']})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(json.dumps({'seed':seed,'training_seconds':result['training_seconds'],'artifact':artifact}),flush=True)


if __name__=='__main__': main()
