"""Role-separated query/record retrieval with frozen pretrained features.

Random episode values prevent stable entity->label lookup. Explicit external label
feedback builds value vectors; no automatic sentence extraction, M teacher or A
learning. Standard nearest/soft attention references, not novel architecture win.
"""
import copy
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from models.feature_readout import FrozenFeatureReadout
from models.segmented_working_read import read_segments
from evaluation.text_checkpoint import load
from evaluation.text_facts import frozen_hash
from evaluation.provenance import capture

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(seed):
    base,task,artifact=load(ROOT/f'experiments/text_runtime/models800/seed{seed}.pt','cpu');base.use_mem=False
    before=frozen_hash(base);tables={}
    for feature in ('e_key','token_mean'):
        extractor=FrozenFeatureReadout(copy.deepcopy(base),feature)
        x=torch.stack([task.inputs[e,t] for e in range(16) for t in range(6)])
        tables[feature]=extractor.features(x).reshape(16,6,-1)
    rng=torch.Generator().manual_seed(seed+101000);metrics={};traces=[]
    for support in (4,8,16):
        for episode in range(32):
            entities=torch.randperm(16,generator=rng)[:support]
            colors=torch.randint(8,(support,),generator=rng)
            labels=task.labels[colors];wrong_labels=task.labels[(colors+1)%8]
            support_templates=torch.randint(2,(support,),generator=rng)
            for query_name,offset in [('seen',2),('held',4)]:
                query_templates=torch.randint(2,(support,),generator=rng)+offset
                permutation=torch.randperm(support,generator=rng)
                # Every observed entity queried; entity indices used only for lookup/scoring,
                # not supplied to read_segments or used to choose record.
                for feature,table in tables.items():
                    q=table[entities,query_templates]
                    keys=table[entities,support_templates]
                    batch_keys=keys[None].expand(support,-1,-1)
                    for intervention,current_labels in [('normal',labels),('wrong',wrong_labels)]:
                        values=base.code[current_labels][None].expand(support,-1,-1)
                        for method,hard in [('nearest',True),('soft',False)]:
                            result=read_segments(q,batch_keys,values,hard=hard)
                            reordered=read_segments(q,batch_keys[:,permutation],values[:,permutation],hard=hard)
                            if not torch.allclose(result['value'],reordered['value'],atol=1e-6,rtol=1e-5):
                                raise AssertionError('record permutation changed retrieved value')
                            logits=F.normalize(result['value'],dim=-1)@base.code.T
                            predicted=logits.argmax(-1)
                            key=f'{support}/{query_name}/{feature}/{method}/{intervention}'
                            item=metrics.setdefault(key,{'n':0,'correct':0,'record_correct':0,'matches_written_value':0})
                            item['n']+=support;item['correct']+=int((predicted==labels).sum())
                            item['record_correct']+=int((result['top_record']==torch.arange(support)).sum())
                            item['matches_written_value']+=int((predicted==current_labels).sum())
                            traces.append({'support':support,'episode':episode,'query':query_name,'feature':feature,
                                'method':method,'intervention':intervention,'entities':entities.tolist(),
                                'support_templates':support_templates.tolist(),'query_templates':query_templates.tolist(),
                                'target_ids':labels.tolist(),'written_ids':current_labels.tolist(),
                                'predicted_ids':predicted.tolist(),'top_record':result['top_record'].tolist(),
                                'permutation':permutation.tolist()})
    for item in metrics.values():
        item['accuracy']=item['correct']/item['n'];item['record_accuracy']=item['record_correct']/item['n']
        item['written_match_rate']=item['matches_written_value']/item['n']
    if before!=frozen_hash(base) or any(m.count_nonzero() for m in base.mem):raise AssertionError('base changed')
    return {'seed':seed,'checkpoint':artifact,'metrics':metrics,'traces':traces,
            'base_unchanged':True,'M_off_zero':True,'record_permutation_value_invariance_checked':True}


def main():
    out=ROOT/'experiments/consolidation/segmented_read.json'
    if out.exists():raise FileExistsError(out)
    report={'scope':'query-key-value separation, standard external retrieval; no new learned routing/Online semantics',
            'design':'32episodes/support4,8,16, fresh random8color mapping each episode, fixed temperature.1',
            'provenance':capture(['models/segmented_working_read.py','models/feature_readout.py','evaluation/segmented_read_control.py']),
            'runs':[]}
    for seed in (0,1,2):
        result=run(seed);report['runs'].append(result)
        print(json.dumps({'seed':seed,'held16':{k:v['accuracy'] for k,v in result['metrics'].items()
                    if k.startswith('16/held') and k.endswith('normal')}}),flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
