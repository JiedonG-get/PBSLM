"""Observed-key dual-write reference with explicitly retained keys and values.

Full-row-rank known key set only. Finite-value records supplied by caller; no semantic
identity discovery/unknown detection. State K,V,D,M retained; NOT history-free or
same-resource as delta fast state. Rebuild/solve at construction, no neural training.
"""
import torch
import torch.nn.functional as F


class DualKeyReference:
    def __init__(self,keys,values):
        k=F.normalize(torch.as_tensor(keys).detach().cpu().double(),dim=-1)
        v=torch.as_tensor(values).detach().cpu().double().clone()
        if k.dim()!=3 or v.dim()!=3 or k.shape[:2]!=v.shape[:2] or k.shape[1]>k.shape[2]:
            raise ValueError('batch of <=key_dim keys and matching value rows required')
        if not torch.isfinite(k).all() or not torch.isfinite(v).all() or (k.norm(dim=-1)<.5).any():
            raise ValueError('finite nonzero keys and finite values required')
        gram=k@k.transpose(1,2)
        eig=torch.linalg.eigvalsh(gram)
        if (eig[:,0]<1e-8).any():raise ValueError('full row rank sufficiently separated keys required')
        self.keys=k.clone();self.values=v
        # D has columns d_i such that K d_i is the i-th unit vector.
        self.dual=torch.linalg.solve(gram,k).transpose(1,2)
        self.matrix=self.dual@self.values
        self.condition=float((eig[:,-1]/eig[:,0]).max())

    def overwrite(self,key_index,value):
        if type(key_index) is not int or not 0<=key_index<self.keys.shape[1]:raise ValueError('known record index required')
        target=torch.as_tensor(value).detach().cpu().double()
        if target.shape!=self.values[:,key_index].shape or not torch.isfinite(target).all():
            raise ValueError('finite matching replacement value required')
        residual=target-self.values[:,key_index]
        updated=self.matrix+self.dual[:,:,key_index,None]*residual[:,None,:]
        if not torch.isfinite(updated).all():raise FloatingPointError('nonfinite dual update')
        self.matrix=updated;self.values[:,key_index]=target

    def read(self,queries):
        q=torch.as_tensor(queries).detach().cpu().double()
        if q.dim()!=3 or q.shape[0]!=self.keys.shape[0] or q.shape[2]!=self.keys.shape[2] or not torch.isfinite(q).all():
            raise ValueError('matching finite batch query matrix required')
        return F.normalize(q,dim=-1)@self.matrix

    def storage_per_stream(self):
        return {name+'_bytes':tensor[0].numel()*tensor.element_size()
                for name,tensor in [('keys',self.keys),('values',self.values),('dual',self.dual),('matrix',self.matrix)]}
