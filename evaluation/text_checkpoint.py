"""Inference-only checkpoint for controlled Chinese text facts; not legacy v2 format."""
import hashlib
import json
from pathlib import Path

import torch
from models.model_v2 import PBSLMv2
from evaluation.text_facts import TextFacts, ENTITIES, COLORS, TEMPLATES, model_config

FORMAT='pbslm.controlled_text_facts.inference.v1'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def save(path,model,task,training,provenance):
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    canonical = model_config(task)
    actual = {name: getattr(model, name) for name in canonical if hasattr(model, name)}
    actual['m_max'] = int(model.q_emb.shape[0])
    actual['e_depth'] = int(model.ebank.depth)
    if actual != canonical:
        raise ValueError('model inference configuration does not match text checkpoint contract')
    # Start a live session from a clean M, not the last evaluation intervention.
    state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    for name,value in state.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'nonfinite checkpoint tensor: {name}')
    for name in state:
        if name.startswith('mem.'): state[name].zero_()
    payload={'format':FORMAT,'model_config':model_config(task),'state_dict':state,
             'task':{'compact_to_bpe_ids':task.original_ids,'tokenizer_sha256':task.tokenizer_hash,
                     'input_length':task.length,'entities':ENTITIES,'colors':COLORS,'templates':TEMPLATES},
             'training':json.loads(json.dumps(training)),'provenance':json.loads(json.dumps(provenance)),
             'note':'inference checkpoint; no optimizer/resume state; M reset to zero'}
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,path)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha256(path)}


def load(path,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('format')!=FORMAT: raise ValueError('unsupported text checkpoint format')
    task=TextFacts(device)
    expected={'compact_to_bpe_ids':task.original_ids,'tokenizer_sha256':task.tokenizer_hash,
              'input_length':task.length,'entities':ENTITIES,'colors':COLORS,'templates':TEMPLATES}
    if ck['task']!=expected: raise ValueError('tokenizer/text inventory contract mismatch')
    if ck['model_config']!=model_config(task): raise ValueError('unsupported text model configuration')
    model=PBSLMv2(**ck['model_config']).to(device).eval()
    if any(not torch.isfinite(value).all() for name,value in ck['state_dict'].items()):
        raise FloatingPointError('nonfinite inference checkpoint tensor')
    # Call base strict loader: no silent historical out_c migration in this format.
    torch.nn.Module.load_state_dict(model,ck['state_dict'],strict=True)
    if any(value.count_nonzero() for value in model.mem): raise ValueError('checkpoint M must be clean')
    return model,task,{'sha256':sha256(path),'training':ck['training'],'provenance':ck['provenance']}
