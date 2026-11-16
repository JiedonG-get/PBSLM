"""Stage-boundary low-rank sufficient-statistic sketch for ridge readout.

Stores U,s and projected cross statistic. Discarded directions cannot be recovered
or undone; no raw slot correction/source semantics. Uses transient dense eigensolve,
so persistent-state savings are not peak-memory or real-time savings.
"""
import torch


class LowRankRidge:
    def __init__(self,features,outputs,rank,*,regularization=.01):
        if any(type(v) is not int or v<1 for v in (features,outputs,rank)) or rank>features+1:
            raise ValueError('positive dimensions and rank<=augmented feature dimension required')
        if isinstance(regularization,bool) or not isinstance(regularization,(int,float)) or not 0<regularization<float('inf'):
            raise ValueError('finite positive regularization required')
        self.features=features;self.outputs=outputs;self.rank=rank;self.regularization=regularization
        self.basis=torch.zeros(features+1,0,dtype=torch.float64)
        self.values=torch.zeros(0,dtype=torch.float64)
        self.projected_cross=torch.zeros(0,outputs,dtype=torch.float64)
        self.samples=0;self.updates=0;self.diagnostics=[]

    def _input(self,x):
        x=torch.as_tensor(x).detach().cpu().double()
        if x.dim()!=2 or x.shape[1]!=self.features or not torch.isfinite(x).all():
            raise ValueError('finite matching features required')
        return torch.cat((x,torch.ones(len(x),1,dtype=x.dtype)),dim=1)

    def add_stage(self,x,y):
        x=self._input(x);y=torch.as_tensor(y).detach().cpu().double()
        if not len(x) or y.shape!=(len(x),self.outputs) or not torch.isfinite(y).all():
            raise ValueError('nonempty finite matching targets required')
        gram=(self.basis*self.values[None,:])@self.basis.T+x.T@x
        cross=self.basis@self.projected_cross+x.T@y
        if not torch.isfinite(gram).all() or not torch.isfinite(cross).all():
            raise FloatingPointError('nonfinite sketch update')
        eigenvectors_values,basis=torch.linalg.eigh((gram+gram.T)/2)
        index=torch.arange(len(eigenvectors_values)-1,max(-1,len(eigenvectors_values)-1-self.rank),-1)
        basis=basis[:,index];values=eigenvectors_values[index].clamp_min(0)
        projected=basis.T@cross
        residual=cross-basis@projected
        diagnostic={'update':self.updates+1,'incoming_rows':len(x),
                    'discarded_gram_trace':max(0.,float(gram.trace()-values.sum())),
                    'discarded_cross_frobenius':float(torch.linalg.norm(residual)),
                    'note':'relative to previously sketched history plus current stage, not original all-history truth'}
        if not torch.isfinite(projected).all():raise FloatingPointError('nonfinite projected statistic')
        self.basis,self.values,self.projected_cross=basis,values,projected
        self.samples+=len(x);self.updates+=1;self.diagnostics.append(diagnostic)

    def predict(self,x):
        augmented=self._input(x)
        return ((augmented@self.basis)/(self.values+self.regularization)[None,:])@self.projected_cross

    def storage(self):
        return {'basis_bytes':self.basis.numel()*8,'eigenvalue_bytes':self.values.numel()*8,
                'projected_cross_bytes':self.projected_cross.numel()*8,
                'tensor_payload_bytes':sum(t.numel()*8 for t in (self.basis,self.values,self.projected_cross)),
                'samples_accumulated':self.samples,'rank_limit':self.rank,
                'raw_rows_retained':0,'full_weight_retained':False,
                'note':'excludes diagnostics/Python/input arrays and dense eigensolve transient workspace'}
