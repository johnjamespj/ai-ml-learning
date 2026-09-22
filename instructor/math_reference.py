"""Numerical answer key. Written proofs are assessed separately."""
import numpy as np

def gradient(X,y,w): return 2*X.T@(X@w-y)/len(y)
def map_coin(heads,tails,alpha,beta):
    a=alpha+heads;b=beta+tails
    if a<=1 or b<=1: raise ValueError('This function is the interior-mode exercise only')
    return (a-1)/(a+b-2)
def kl(p,q):
    p=np.asarray(p);q=np.asarray(q)
    positive=p>0
    if (q[positive]==0).any(): return np.inf
    return float(np.sum(p[positive]*np.log(p[positive]/q[positive])))
def principal_direction(centered_X): return np.linalg.svd(centered_X,full_matrices=False)[2][0]
def affine_backward(activations,upstream): return activations.T@upstream,upstream.sum(0)
def stable_softmax(scores):
    z=scores-scores.max(-1,keepdims=True);p=np.exp(z);return p/p.sum(-1,keepdims=True)
def bellman_backup(V,P,R,gamma): return (P*(R+gamma*V[None,None,:])).sum(-1).max(1)
def colored_matched_score(x,s,C): return s@np.linalg.solve(C,x)
