"""Fixed-capacity dynamic content-slot reference with frozen learned value reader.

No entity IDs; exact query vector drives matching. Explicit keys retained in tensor
state, NOT key-free compressed neural memory or a novel allocation algorithm.
Fixed cosine threshold and update-recency eviction. All slot candidates allocated
within fixed state; no raw event history. State rows: key16,value8,occupied,writeclock.
"""
import torch
from torch import nn
import torch.nn.functional as F


class DynamicContentSlots(nn.Module):
    def __init__(self,fast_model,capacity=8,threshold=.9):
        super().__init__()
        if (type(capacity) is not int or capacity<1 or isinstance(threshold,bool)
                or not isinstance(threshold,(int,float)) or not 0<threshold<=1):
            raise ValueError('positive capacity and threshold in(0,1] required')
        if fast_model.key_dim!=16 or fast_model.values!=8:raise ValueError('canonical value reader required')
        self.reader=fast_model;self.capacity=capacity;self.threshold=threshold
        for p in self.reader.parameters():p.requires_grad_(False)
        self.eval()

    def initial(self,batch):return self.reader.value_embedding.weight.new_zeros(batch,self.capacity,26)

    def _match(self,state,key):
        if (state.dim()!=3 or state.shape[1:]!=(self.capacity,26) or key.shape!=(len(state),16)
                or not torch.isfinite(state).all() or not torch.isfinite(key).all()
                or (key.norm(dim=-1)<1e-8).any()):
            raise ValueError('finite state and matching nonzero16D key batch required')
        normalized=F.normalize(key,dim=-1)
        score=torch.einsum('bd,bkd->bk',normalized,F.normalize(state[:,:,:16],dim=-1))
        score=score.masked_fill(state[:,:,24]<.5,-2.)
        best,index=score.max(-1)
        return normalized,best,index

    @torch.no_grad()
    def observe(self,state,key,value):
        k,best,index=self._match(state,key)
        if value.shape!=(len(state),) or value.dtype!=torch.int64 or (value<0).any() or (value>=8).any():
            raise ValueError('matching integer class0..7 batch required')
        new=state.clone();targets=self.reader.value_embedding(value)
        for b in range(len(state)):
            occupied=state[b,:,24]>.5
            match=bool(best[b]>=self.threshold)
            if match:slot=int(index[b])
            elif (~occupied).any():slot=int((~occupied).nonzero()[0])
            else:slot=int(state[b,:,25].argmin())
            # Full latest value replacement; no leak into other slots.
            new[b,slot,:16]=k[b]
            new[b,slot,16:24]=targets[b]
            new[b,slot,24]=1.
            # Bounded recency ordering, not an unbounded float timestamp.
            old_rank=state[b,slot,25] if occupied[slot] else state.new_tensor(-1.)
            ranks=state[b,:,25].clone()
            if occupied[slot]:ranks[(occupied)&(ranks>old_rank)]-=1.
            new[b,:,25]=ranks
            new[b,slot,25]=float(occupied.sum()-int(occupied[slot]))
        return new

    def query(self,state,key):
        _,best,index=self._match(state,key);rows=torch.arange(len(state),device=state.device)
        logits=self.reader.read(state[rows,index,16:24])
        unknown=best<self.threshold
        explicit=torch.full_like(logits,-30.);explicit[:,8]=30.
        return torch.where(unknown[:,None],explicit,logits)
