"""Frozen-model write-gating interventions, read routing unchanged.

Initial priming always original learned soft rule. Only subsequent hot writes use
soft/top1 original-mass/top1 unit gates. This is test-time distribution shift, not a
trained sparse architecture or proof that every failure is due to gate leakage.
"""
import torch


@torch.no_grad()
def observe_with_gate(model,state,entities,values,mode):
    if mode not in ('soft','top1_mass','top1_unit'):raise ValueError('unsupported write intervention')
    if model.addressing!='learned':raise ValueError('learned-route checkpoint required')
    inputs=torch.cat((model.entity_embedding(entities),model.value_embedding(values)),-1)
    candidate=model.cell(inputs[:,None,:].expand(-1,model.slots,-1).reshape(-1,48),
                         state.reshape(-1,model.slot_width)).reshape_as(state)
    route=model.routing(entities)
    if mode!='soft':
        top=route.argmax(-1,keepdim=True)
        retained=route.gather(1,top) if mode=='top1_mass' else torch.ones_like(route.gather(1,top))
        route=torch.zeros_like(route).scatter_(1,top,retained)
    return state+route[:,:,None]*(candidate-state)
