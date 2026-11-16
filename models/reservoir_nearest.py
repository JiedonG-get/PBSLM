"""Feature/label reservoir nearest-neighbor reference with tensor-payload budget.

Rows float64, labels int64; shared codebook/base excluded for all methods. Reservoir
counter/RNG/Python metadata not included, so NOT a whole-process equal-memory claim.
"""
import random
import torch
import torch.nn.functional as F


class ReservoirNearest:
    def __init__(self,features,budget,*,seed=0):
        if type(features) is not int or features<1 or type(budget) is not int:
            raise ValueError('integer dimension and budget required')
        self.features=features;self.row_bytes=features*8+8
        self.capacity=budget//self.row_bytes
        if self.capacity<1:raise ValueError('budget cannot retain one feature/label row')
        self.x=torch.zeros(self.capacity,features,dtype=torch.float64)
        self.y=torch.zeros(self.capacity,dtype=torch.int64)
        self.seen=0;self.used=0;self.rng=random.Random(seed)

    def add(self,x,y):
        x=torch.as_tensor(x).detach().cpu().double();y=torch.as_tensor(y).detach().cpu()
        if (x.dim()!=2 or x.shape[1]!=self.features or y.shape!=(len(x),) or y.dtype!=torch.int64
                or not torch.isfinite(x).all() or (y<0).any()):raise ValueError('finite matching feature/label rows required')
        for row,label in zip(x,y):
            self.seen+=1
            slot=self.used if self.used<self.capacity else self.rng.randrange(self.seen)
            if slot<self.capacity:self.x[slot].copy_(row);self.y[slot]=label
            self.used=min(self.used+1,self.capacity)

    def predict(self,x):
        if not self.used:raise ValueError('empty reservoir')
        x=torch.as_tensor(x).detach().cpu().double()
        if x.dim()!=2 or x.shape[1]!=self.features or not torch.isfinite(x).all():raise ValueError('finite features required')
        nearest=(F.normalize(x,dim=-1)@F.normalize(self.x[:self.used],dim=-1).T).argmax(-1)
        return self.y[:self.used][nearest]

    def storage(self):
        return {'capacity_rows':self.capacity,'used_rows':self.used,'seen_rows':self.seen,
                'tensor_payload_bytes':self.x.numel()*8+self.y.numel()*8,
                'note':'allocated feature/label tensors only, excludes RNG/counter/Python/base/code'}
