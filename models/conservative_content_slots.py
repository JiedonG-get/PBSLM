"""Explicit anchored-key, no-eviction admission reference, not learned safety.

Unmatched writes at full capacity are returned as deferred, NOT queued/stored. Caller
must retry or discard knowingly. Similar different identities may still merge.
"""
import torch
from .dynamic_content_slots import DynamicContentSlots


class ConservativeContentSlots(DynamicContentSlots):
    @torch.no_grad()
    def observe_receipt(self,state,key,value):
        normalized,best,index=self._match(state,key)
        if value.shape!=(len(state),) or value.dtype!=torch.int64 or (value<0).any() or (value>=8).any():
            raise ValueError('matching integer class0..7 batch required')
        new=state.clone();targets=self.reader.value_embedding(value)
        accepted=torch.zeros(len(state),dtype=torch.bool,device=state.device)
        for b in range(len(state)):
            occupied=state[b,:,24]>.5;matched=bool(best[b]>=self.threshold)
            if matched:slot=int(index[b])
            elif (~occupied).any():slot=int((~occupied).nonzero()[0])
            else:continue
            accepted[b]=True
            # Anchor remains original observed key after a match.
            if not matched:new[b,slot,:16]=normalized[b]
            new[b,slot,16:24]=targets[b];new[b,slot,24]=1.
            old=state[b,slot,25] if occupied[slot] else state.new_tensor(-1.)
            ranks=state[b,:,25].clone()
            if occupied[slot]:ranks[occupied&(ranks>old)]-=1.
            new[b,:,25]=ranks;new[b,slot,25]=float(occupied.sum()-int(occupied[slot]))
        return new,{'accepted':accepted,'deferred':~accepted,'best_cosine':best.detach().clone()}

    def observe(self,state,key,value):
        return self.observe_receipt(state,key,value)[0]
