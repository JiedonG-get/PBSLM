"""Controlled Chinese template-fact recall with observed support and paraphrase split.

Uses existing repository BPE then a recorded compact ID table for a small model.
Not full-vocabulary language modeling. Each episode randomizes entity colors. Only
support answers write M; eval parameters never update. Familiar/novel entities and
seen/held-out question templates are reported separately.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from models.model_v2 import PBSLMv2
from evaluation.provenance import capture

ROOT = Path(__file__).resolve().parents[1]
ENTITIES = [x + '号设备' for x in '甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳']
COLORS = ['红', '蓝', '绿', '黄', '黑', '白', '紫', '灰']
TEMPLATES = ['{entity}的颜色是什么？', '请说出{entity}的颜色。',
             '请回答{entity}是什么颜色。', '{entity}是什么颜色的？',
             '我想知道{entity}的颜色。', '告诉我，{entity}的颜色是什么？']


class TextFacts:
    def __init__(self, device):
        path = ROOT / 'tokenizer/bpe16k_tokenizer.json'
        self.tokenizer_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        tokenizer = Tokenizer.from_file(str(path))
        strings = [template.format(entity=e) for e in ENTITIES for template in TEMPLATES]
        ids = {s: tokenizer.encode(s).ids for s in strings + COLORS}
        if any(len(ids[c]) != 1 for c in COLORS):
            raise ValueError('Color answers must be single BPE tokens')
        physical = sorted({0} | {token for row in ids.values() for token in row})
        mapping = {token: index for index, token in enumerate(physical)}
        self.original_ids = physical
        self.mapping = mapping
        self.vocab = len(physical)
        self.length = max(len(row) for row in ids.values())
        self.device = device
        self.pad_id = mapping[0]
        self.inputs = {}
        for e, entity in enumerate(ENTITIES):
            for t, template in enumerate(TEMPLATES):
                row = ids[template.format(entity=entity)]
                self.inputs[e, t] = torch.tensor([self.pad_id] * (self.length-len(row)) +
                                                 [mapping[x] for x in row], device=device)
        self.labels = torch.tensor([mapping[ids[c][0]] for c in COLORS], device=device)
        self.color_tokens = {int(mapping[ids[c][0]]): c for c in COLORS}
        self.original_bpe_by_text = ids
        input_inventory = {int(x) for e in range(12) for t in range(4) for x in self.inputs[e,t]}
        query_gradient_inventory = {int(x) for e in range(12) for t in (2,3) for x in self.inputs[e,t]}
        novel = {int(x) for e in range(12,16) for t in range(6) for x in self.inputs[e,t]}
        self.token_inventory_evidence = {
            'transductive_vocab_and_padding': True,
            'novel_split_tokens_absent_from_training_input_inventory': sorted(novel - input_inventory),
            'novel_split_tokens_absent_from_main_objective_query_inventory': sorted(novel - query_gradient_inventory),
            'note': 'auxiliary key alignment also backpropagates support features when enabled; input occurrence alone does not prove a trained embedding'}

    def episode(self, gen, split, support=4):
        if split not in ('familiar', 'novel') or type(support) is not int or not 2 <= support <= 4:
            raise ValueError('split must be familiar/novel and support integer 2..4')
        pool = torch.arange(12) if split == 'familiar' else torch.arange(12,16)
        entities = pool[torch.randperm(len(pool), generator=gen)[:support]].tolist()
        colors = torch.randperm(8, generator=gen)[:support].tolist()
        support_templates = torch.randint(0,2,(support,),generator=gen).tolist()
        query_templates = torch.randint(2,4,(support,),generator=gen).tolist()
        held_templates = torch.randint(4,6,(support,),generator=gen).tolist()
        sx = torch.stack([self.inputs[e,t] for e,t in zip(entities,support_templates)])
        qx = torch.stack([self.inputs[e,t] for e,t in zip(entities,query_templates)])
        hx = torch.stack([self.inputs[e,t] for e,t in zip(entities,held_templates)])
        y = self.labels[torch.tensor(colors,device=self.device)]
        return sx, qx, hx, y, {'entities':entities,'colors':colors,'support_templates':support_templates,
                              'query_templates':query_templates,'held_templates':held_templates}


@torch.no_grad()
def fill(model, sx, y, mode):
    for mem in model.mem:
        mem.zero_()
    if mode == 'zero':
        return
    values = y.roll(1) if mode == 'wrong' else y
    for row,label in zip(sx,values):
        features,_ = model.e_forward(model.token_emb(row[None]))
        model.write_memory(features,model.target_embed(label.reshape(1,1)),lr=.05)


def frozen_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if not name.startswith('mem.'):
            h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, task, episodes, seed):
    before = frozen_hash(model)
    result = {}
    traces = []
    for split in ('familiar','novel_entities'):
        gen=torch.Generator().manual_seed(seed+(0 if split=='familiar' else 1))
        sums={}
        for index in range(episodes):
            sx,qx,hx,y,meta=task.episode(gen,'familiar' if split=='familiar' else 'novel')
            c_reference={}
            for mode in ('normal','zero','wrong'):
                fill(model,sx,y,mode)
                for query_name,x in [('same_surface',sx),('seen_paraphrase',qx),('held_paraphrase',hx)]:
                    out=model(x,1,sample_noise=False)
                    pa=model.decode_nearest(out['z_a']).argmax(-1)[:,0]
                    pc=model.decode_nearest(out['z_c']).argmax(-1)[:,0]
                    if query_name in c_reference and not torch.equal(c_reference[query_name],out['z_c']):
                        raise AssertionError('Memory changed C')
                    c_reference[query_name]=out['z_c'].clone()
                    k=f'{split}/{query_name}/{mode}'
                    item=sums.setdefault(k,{'correct':0,'c_correct':0,'n':0,'written_matches':0,
                                            'unique_predictions_sum':0,'episodes':0})
                    item['unique_predictions_sum']+=len(pa.unique()); item['episodes']+=1
                    item['correct']+=int((pa==y).sum()); item['c_correct']+=int((pc==y).sum())
                    item['written_matches']+=int((pa==(y.roll(1) if mode=='wrong' else y)).sum())
                    item['n']+=len(y)
                    traces.append({'split':split,'episode':index,'mode':mode,'query':query_name,
                                   **meta,'target_ids':y.tolist(),'a_ids':pa.tolist(),'c_ids':pc.tolist()})
        for key,item in sums.items():
            item['accuracy']=item['correct']/item['n']; item['c_accuracy']=item['c_correct']/item['n']
            item['written_match_rate']=item['written_matches']/item['n']
            item['mean_unique_predictions_per_episode']=item['unique_predictions_sum']/item['episodes']
        result.update(sums)
    if frozen_hash(model)!=before:
        raise AssertionError('Evaluation changed weights/buffers')
    return {'metrics':result,'traces':traces,'weights_unchanged':True,'c_memory_invariant':True}


def model_config(task):
    return dict(vocab=task.vocab,k=8,depth=1,d=48,heads=4,m_max=2,act_x=.5,
                e_n=2,e_depth=1,emo_dim=8,refine=1,field_scale=0.,
                trainable_target=False,use_mem=True,use_flow=True,
                simple_dec=False,dense=False,direct_only=False,use_sample_masks=True)


def run(seed,args,return_model=False):
    torch.manual_seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    task=TextFacts(device)
    model=PBSLMv2(**model_config(task)).to(device).eval()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.002)
    gen=torch.Generator().manual_seed(seed+7100)
    curve=[]; start=time.perf_counter()
    for step in range(args.episodes):
        sx,qx,hx,y,meta=task.episode(gen,'familiar')
        fill(model,sx,y,'normal')
        # Training uses query templates 2/3, never held templates 4/5 or novel entities.
        out=model(qx,1,sample_noise=False)
        logits=out['z_a'][:,0].float()@model.code.float().t()/.1
        objective=F.cross_entropy(logits,y)
        alignment = torch.zeros((), device=device)
        if args.key_align_weight > 0:
            support_features, _ = model.e_forward(model.token_emb(sx))
            query_features, _ = model.e_forward(model.token_emb(qx))
            # Explicit entity-identity supervision across templates; no color label in keys.
            support_keys = F.normalize(support_features.mean(2), dim=-1)   # (E,B,D)
            query_keys = F.normalize(query_features.mean(2), dim=-1)
            scores = torch.matmul(query_keys, support_keys.transpose(-1,-2)) / .1
            targets = torch.arange(len(y), device=device).expand(model.e_n,-1)
            alignment = F.cross_entropy(scores.reshape(-1,len(y)), targets.reshape(-1))
            objective = objective + args.key_align_weight * alignment
        if not torch.isfinite(objective): raise FloatingPointError('nonfinite objective')
        opt.zero_grad(set_to_none=True); objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step()
        if (step+1)%100==0:
            curve.append({'step':step+1,'loss':float(objective.detach()),
                          'key_alignment_loss':float(alignment.detach()),
                          'accuracy':float((logits.argmax(-1)==y).float().mean())})
    training_seconds=time.perf_counter()-start
    evaluation=evaluate(model,task,args.eval_episodes,seed+8100)
    result = {'seed':seed,'vocab':task.vocab,'compact_to_bpe_ids':task.original_ids,
            'tokenizer_sha256':task.tokenizer_hash,'padded_input_length':task.length,
            'token_inventory_evidence':task.token_inventory_evidence,
            'original_bpe_by_text':task.original_bpe_by_text,
            'curve':curve,'training_seconds':training_seconds,'total_seconds':time.perf_counter()-start,
            'evaluation':evaluation}
    return (model,task,result) if return_model else result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=[0]); p.add_argument('--episodes',type=int,default=800)
    p.add_argument('--eval-episodes',type=int,default=32)
    p.add_argument('--key-align-weight',type=float,default=0., help='Explicit entity-identity contrastive key objective; 0 = original task')
    p.add_argument('--output',default='experiments/text_facts/pilot.json'); args=p.parse_args()
    if args.episodes < 1 or args.eval_episodes < 1 or args.key_align_weight < 0:
        p.error('positive episode counts and nonnegative alignment weight required')
    out=ROOT/args.output
    if out.exists(): raise FileExistsError(out)
    report={'config':vars(args),'entities':ENTITIES,'colors':COLORS,'templates':TEMPLATES,
            'provenance':capture(['evaluation/text_facts.py','models/model.py','models/model_v2.py']),
            'scope':'controlled Chinese template recall, compact BPE sub-vocabulary, not full language model','runs':[]}
    for seed in args.seeds:
        result=run(seed,args); report['runs'].append(result)
        print(json.dumps({'seed':seed,'seconds':result['total_seconds'],'vocab':result['vocab'],
                          'metrics':result['evaluation']['metrics']},ensure_ascii=False),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__': main()
