"""Fixed-size observed-key span novelty reference; no raw key list.

P projects orthogonally onto complement of observed key span. Small residual means
IN SPAN, not 'this exact identity was observed'. At full rank P vanishes and novelty
cannot reject anything. Stores D²float64, extra to fast-state value matrix.
"""
import torch
import torch.nn.functional as F


class KeySpanState:
    def __init__(self,dimension=16):
        if type(dimension) is not int or dimension<1:raise ValueError('positive dimension required')
        self.dimension=dimension;self.projector=torch.eye(dimension,dtype=torch.float64);self.rank=0

    def _key(self,key):
        key=torch.as_tensor(key).detach().cpu().double()
        if key.shape!=(self.dimension,) or not torch.isfinite(key).all() or key.norm()<1e-10:
            raise ValueError('finite nonzero key vector required')
        return F.normalize(key,dim=0)

    def observe(self,key):
        k=self._key(key);residual=self.projector@k;energy=residual@residual
        if energy>1e-12 and self.rank<self.dimension:
            p=self.projector-torch.outer(residual,residual)/energy
            p=(p+p.T)/2
            if not torch.isfinite(p).all():raise FloatingPointError('nonfinite span state')
            self.projector=p;self.rank+=1

    def novelty(self,queries):
        q=torch.as_tensor(queries).detach().cpu().double()
        if q.dim()!=2 or q.shape[1]!=self.dimension or not torch.isfinite(q).all() or (q.norm(dim=-1)<1e-10).any():
            raise ValueError('finite nonzero query rows required')
        return torch.linalg.vector_norm(F.normalize(q,dim=-1)@self.projector.T,dim=-1)
