"""Cumulative ridge sufficient-statistic readout research reference.

Not original A optimizer, not zero-memory storage: retains O(D²+D*V) aggregate
constraints. CPU float64 solves at explicit consolidation boundaries, not per-token
low-latency inference. No stored raw examples, no deletion/correction support yet.
"""
import torch


class CumulativeRidge:
    def __init__(self,features,outputs,*,regularization=.01):
        if type(features) is not int or type(outputs) is not int or min(features,outputs)<1:
            raise ValueError('positive dimensions required')
        if not isinstance(regularization,(int,float)) or isinstance(regularization,bool) or not 0<regularization<float('inf'):
            raise ValueError('finite positive regularization required')
        self.features=features;self.outputs=outputs;self.regularization=regularization
        self.gram=torch.zeros(features+1,features+1,dtype=torch.float64)
        self.cross=torch.zeros(features+1,outputs,dtype=torch.float64)
        self.weight=torch.zeros(features+1,outputs,dtype=torch.float64)
        self.samples=0

    def _features(self,x):
        x=torch.as_tensor(x).detach().to(device='cpu',dtype=torch.float64)
        if x.dim()!=2 or x.shape[1]!=self.features or not torch.isfinite(x).all():
            raise ValueError('finite feature matrix required')
        return torch.cat((x,torch.ones(x.shape[0],1,dtype=x.dtype)),dim=1)

    def add(self,x,y):
        x=self._features(x);y=torch.as_tensor(y).detach().to(device='cpu',dtype=torch.float64)
        if y.shape!=(x.shape[0],self.outputs) or not torch.isfinite(y).all() or not len(x):
            raise ValueError('nonempty finite matching target matrix required')
        new_gram=self.gram+x.t()@x;new_cross=self.cross+x.t()@y
        if not torch.isfinite(new_gram).all() or not torch.isfinite(new_cross).all():
            raise FloatingPointError('nonfinite sufficient statistics')
        self.gram,self.cross=new_gram,new_cross;self.samples+=len(x)

    def solve(self):
        # Bias regularized too, recorded as part of objective ||XW-Y||²+lambda||W||².
        system=self.gram+self.regularization*torch.eye(self.features+1,dtype=torch.float64)
        weight=torch.linalg.solve(system,self.cross)
        if not torch.isfinite(weight).all():raise FloatingPointError('nonfinite ridge solution')
        self.weight=weight
        return weight.clone()

    def predict(self,x):return self._features(x)@self.weight

    def storage(self):
        return {'gram_bytes':self.gram.numel()*self.gram.element_size(),
                'cross_bytes':self.cross.numel()*self.cross.element_size(),
                'weight_bytes':self.weight.numel()*self.weight.element_size(),
                'samples_accumulated':self.samples,'raw_examples_stored':0,
                'note':'excludes Python/model/features/code and transient solve workspace'}
