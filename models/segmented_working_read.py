"""Explicit query/record-key/value separation, standard cosine retrieval reference.

This is external working-memory routing, not original A weight consolidation or a
new attention algorithm. Hard argmax ties resolve by record order; soft readout is
permutation-invariant up to floating precision. Keys/values are supplied observations/feedback, not facts
verified or extracted from arbitrary text. No record update/persistence API here.
"""
import torch
import torch.nn.functional as F


def read_segments(query,keys,values,*,temperature=.1,hard=False):
    if (query.dim()!=2 or keys.dim()!=3 or values.dim()!=3 or keys.shape[:2]!=values.shape[:2]
            or query.shape[0]!=keys.shape[0] or query.shape[1]!=keys.shape[2]
            or keys.shape[1]<1 or values.shape[2]<1):
        raise ValueError('query BxF, nonempty keys BxKxF, values BxKxV required')
    if isinstance(temperature,bool) or not isinstance(temperature,(int,float)) or not 0<temperature<float('inf'):
        raise ValueError('finite positive temperature required')
    if any(not torch.isfinite(x).all() for x in (query,keys,values)):
        raise ValueError('finite query, keys and values required')
    scores=torch.einsum('bf,bkf->bk',F.normalize(query,dim=-1),F.normalize(keys,dim=-1))
    top=scores.argmax(-1)
    if hard:
        weights=torch.zeros_like(scores).scatter_(1,top[:,None],1.)
    else:weights=(scores/temperature).softmax(-1)
    output=torch.einsum('bk,bkv->bv',weights,values)
    return {'value':output,'weights':weights,'top_record':top,'cosine_scores':scores}
