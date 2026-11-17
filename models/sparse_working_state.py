"""Matched-training sparse writes; straight-through soft gradient is a surrogate.

Static8-ID routing, soft query unchanged. Collision prior over all known entity IDs
is extra structural supervision, not dynamic entity discovery or language ability.
"""
import torch
import torch.nn.functional as F
from .slotted_working_state import SlottedWorkingState


class SparseWorkingState(SlottedWorkingState):
    def observe(self,state,entities,values):
        inputs=torch.cat((self.entity_embedding(entities),self.value_embedding(values)),-1)
        candidate=self.cell(inputs[:,None,:].expand(-1,self.slots,-1).reshape(-1,48),
                            state.reshape(-1,self.slot_width)).reshape_as(state)
        soft=self.routing(entities)
        hard=F.one_hot(soft.argmax(-1),self.slots).to(soft.dtype)
        weights=hard+soft-soft.detach() if self.training else hard
        return state+weights[:,:,None]*(candidate-state)

    def collision_loss(self):
        route=self.routing(torch.arange(self.entities,device=self.keys.device))
        overlap=route@route.T
        mask=~torch.eye(self.entities,device=self.keys.device,dtype=torch.bool)
        return overlap[mask].mean()
