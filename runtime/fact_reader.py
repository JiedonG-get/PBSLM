"""Supervised candidate extractor for one explicitly bounded entity/color relation.

Separate GRU reader, NOT PBSLM architecture gain. Uses repository full BPE IDs.
A positive classifier is not fact verification. Candidates require external confirm.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
from tokenizers import Tokenizer
from evaluation.text_facts import ENTITIES,COLORS

ROOT=Path(__file__).resolve().parents[1]
MAX_LENGTH=64


class CandidateReader(nn.Module):
    def __init__(self,vocab=16384,embedding=32,hidden=48):
        super().__init__()
        self.embedding=nn.Embedding(vocab,embedding,padding_idx=0)
        self.encoder=nn.GRU(embedding,hidden,batch_first=True,bidirectional=True)
        self.valid=nn.Linear(hidden*2,2)
        self.entity=nn.Linear(hidden*2,len(ENTITIES))
        self.color=nn.Linear(hidden*2,len(COLORS))

    def forward(self,ids,lengths):
        packed=nn.utils.rnn.pack_padded_sequence(self.embedding(ids),lengths.cpu(),
                                                batch_first=True,enforce_sorted=False)
        _,hidden=self.encoder(packed)
        state=torch.cat((hidden[-2],hidden[-1]),dim=-1)
        return self.valid(state),self.entity(state),self.color(state)


def tokenizer():
    path=ROOT/'tokenizer/bpe16k_tokenizer.json'
    return Tokenizer.from_file(str(path)),hashlib.sha256(path.read_bytes()).hexdigest()


def encode(texts,tok,device):
    rows=[tok.encode(text).ids for text in texts]
    if any(not row or len(row)>MAX_LENGTH or 0 in row for row in rows):
        raise ValueError('unsupported empty, unknown-token or overlength fact input')
    lengths=torch.tensor([len(row) for row in rows])
    ids=torch.zeros(len(rows),int(lengths.max()),dtype=torch.long,device=device)
    for b,row in enumerate(rows):ids[b,:len(row)]=torch.tensor(row,device=device)
    return ids,lengths


def load_reader(path,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    tok,digest=tokenizer()
    if (ck.get('format')!='pbslm.fact_reader.v1' or ck['tokenizer_sha256']!=digest
            or ck['entities']!=ENTITIES or ck['colors']!=COLORS):
        raise ValueError('reader inventory/tokenizer contract mismatch')
    model=CandidateReader(**ck['config']).to(device).eval()
    if any(not torch.isfinite(value).all() for value in ck['state_dict'].values()):
        raise FloatingPointError('nonfinite candidate reader checkpoint')
    model.load_state_dict(ck['state_dict'],strict=True)
    return model,tok,{'threshold':ck['threshold'],'margin':ck['margin'],
                     'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}


@torch.no_grad()
def infer(model,texts,tok,device,threshold=.9,margin=.2):
    ids,lengths=encode(texts,tok,device)
    valid,entity,color=model(ids,lengths)
    vp=valid.softmax(-1)[:,1]; ep=entity.softmax(-1); cp=color.softmax(-1)
    e_top=ep.topk(2,dim=-1);c_top=cp.topk(2,dim=-1)
    rows=[]
    for b,text in enumerate(texts):
        e=int(e_top.indices[b,0]);c=int(c_top.indices[b,0])
        accept=(float(vp[b])>=threshold and float(e_top.values[b,0]-e_top.values[b,1])>=margin
                and float(c_top.values[b,0]-c_top.values[b,1])>=margin)
        rows.append({'source_text':text,'source_sha256':hashlib.sha256(text.encode()).hexdigest(),
                     'accepted':accept,'entity':e,'color':c,'entity_name':ENTITIES[e],
                     'color_name':COLORS[c],'valid_score':float(vp[b]),
                     'entity_margin':float(e_top.values[b,0]-e_top.values[b,1]),
                     'color_margin':float(c_top.values[b,0]-c_top.values[b,1]),
                     'warning':'scores uncalibrated; acceptance is extraction policy, not verified truth'})
    return rows
