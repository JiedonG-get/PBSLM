"""Strict loader for existing symbolic working-state research checkpoint v1.

v1 config is partial: only canonical default dimensions supported, not arbitrary
future architectures. Loader checks finite state, exact keys and raw strict shapes.
"""
import hashlib
from pathlib import Path
import torch
from models.recurrent_working_state import RecurrentWorkingState
from models.slotted_working_state import SlottedWorkingState


def load_working_state(path,device='cpu'):
    path=Path(path);payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format')!='pbslm.symbolic_working_state.v1':raise ValueError('unknown working-state format')
    expected={'format','architecture','seed','state_dict','config','updates'}
    if set(payload)!=expected or payload['config']!={'entities':8,'values':8,'state_floats':96}:
        raise ValueError('unsupported partial v1 configuration')
    kind=payload['architecture']
    if kind not in ('global','learned','explicit'):raise ValueError('unknown working-state architecture')
    if type(payload['seed']) is not int or type(payload['updates']) is not int or payload['updates']<1:
        raise ValueError('invalid research metadata')
    state=payload['state_dict']
    if not isinstance(state,dict) or any(not isinstance(t,torch.Tensor) or not torch.isfinite(t).all() for t in state.values()):
        raise ValueError('finite tensor state dictionary required')
    model=RecurrentWorkingState() if kind=='global' else SlottedWorkingState(addressing=kind)
    model.load_state_dict(state,strict=True);model.to(device).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model,{'architecture':kind,'seed':payload['seed'],'updates':payload['updates'],
                 'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'format':payload['format'],'config':payload['config']}
