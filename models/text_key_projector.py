"""Small learned key projector; no color supervision or PBSLM weight updates.

Fixed16D output, identity-contrastive support/query training on known entities only.
Extra entity-identity supervision is explicit, not unsupervised/general semantics.
"""
import torch
from torch import nn
import torch.nn.functional as F


class TextKeyProjector(nn.Module):
    def __init__(self,weight):
        super().__init__()
        if weight.shape!=(96,16) or not torch.isfinite(weight).all():raise ValueError('finite96x16 initialization required')
        self.weight=nn.Parameter(weight.detach().clone())

    def forward(self,features):return F.normalize(features@self.weight,dim=-1)


def key_alignment_loss(support,query):
    if support.dim()!=2 or support.shape!=query.shape or len(support)<2:
        raise ValueError('paired NxD support/query with N>=2 required')
    targets=torch.arange(len(support),device=support.device)
    scores=query@support.T/.1
    identity=.5*(F.cross_entropy(scores,targets)+F.cross_entropy(scores.T,targets))
    # Reduce overlap for different entities while keeping identity alignment objective explicit.
    eye=torch.eye(len(support),device=support.device,dtype=support.dtype)
    geometry=.5*((support@support.T-eye).square().mean()+(query@query.T-eye).square().mean())
    return identity+.1*geometry,{'identity':identity.detach(),'geometry':geometry.detach()}
