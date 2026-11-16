"""Iterative SPD ridge solve from the SAME cumulative normal equations.

Conjugate gradient is an optimization reference, NOT stochastic minibatch SGD.
Uses full historical sufficient statistics and float64, so no memory-free claim.
"""
import torch


def solve_cg(gram,cross,regularization=.01,*,initial=None,max_iterations=500,tolerance=1e-9):
    if gram.dim()!=2 or gram.shape[0]!=gram.shape[1] or cross.dim()!=2 or cross.shape[0]!=gram.shape[0]:
        raise ValueError('matching matrix shapes required')
    if not torch.isfinite(gram).all() or not torch.isfinite(cross).all():raise ValueError('finite matrices required')
    if not isinstance(regularization,(int,float)) or not 0<regularization<float('inf'):
        raise ValueError('positive finite regularization required')
    if type(max_iterations) is not int or max_iterations<1 or not 0<tolerance<float('inf'):
        raise ValueError('positive iteration budget and tolerance required')
    a=gram.detach().cpu().double()+regularization*torch.eye(gram.shape[0],dtype=torch.float64)
    b=cross.detach().cpu().double()
    if not torch.allclose(a,a.T,atol=1e-10,rtol=1e-10):raise ValueError('symmetric gram required')
    x=torch.zeros_like(b) if initial is None else initial.detach().cpu().double().clone()
    if x.shape!=b.shape or not torch.isfinite(x).all():raise ValueError('finite matching initial solution required')
    r=b-a@x;p=r.clone();norm=(r*r).sum();scale=max(float(torch.linalg.norm(b)),1.)
    iterations=0
    for i in range(max_iterations):
        if float(norm.sqrt())<=tolerance*scale:break
        ap=a@p;denominator=(p*ap).sum()
        if not torch.isfinite(denominator) or denominator<=0:raise FloatingPointError('operator not positive definite')
        alpha=norm/denominator;x=x+alpha*p;r=r-alpha*ap
        next_norm=(r*r).sum();iterations=i+1
        p=r+(next_norm/norm)*p;norm=next_norm
    residual=float(torch.linalg.norm(b-a@x))/scale
    if not torch.isfinite(x).all():raise FloatingPointError('nonfinite iterative solution')
    return x,{'iterations':iterations,'relative_residual':residual,'converged':residual<=tolerance*10,
              'max_iterations':max_iterations,'tolerance':tolerance}
