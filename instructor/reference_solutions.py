"""Reference answers. Do not import these while attempting the assessment."""
import math
import numpy as np


def mse_gradient(X,y,w):
    return 2*np.asarray(X).T@(np.asarray(X)@w-y)/len(y)


def posterior_positive(prevalence,pd,pfa):
    if any(not 0<=x<=1 for x in [prevalence,pd,pfa]): raise ValueError('Probabilities must be in [0,1]')
    denom=pd*prevalence+pfa*(1-prevalence)
    if denom==0: raise ValueError('Conditioning event has probability zero')
    return pd*prevalence/denom


def build_pipeline():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))


def logistic_gradient(X,y,w):
    p=np.exp(-np.logaddexp(0,-(X@w)))
    return X.T@(p-y)/len(y)


def relu_backward(upstream,z): return upstream*(z>0)


def cross_entropy_logits(logits,y):
    z=logits-logits.max(1,keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(1))-z[np.arange(len(y)),y]))


def train_step(model,X,y,optimizer):
    from torch import nn
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=nn.CrossEntropyLoss()(model(X),y);loss.backward();optimizer.step()
    return float(loss.detach())


def evaluate_accuracy(model,X,y):
    import torch
    mode=model.training
    model.eval()
    with torch.no_grad(): result=float((model(X).argmax(1)==y).float().mean())
    model.train(mode)
    return result


def attention(Q,K,V,allow_mask):
    allow=np.asarray(allow_mask,bool)
    if not allow.any(-1).all(): raise ValueError('Fully masked row')
    scores=Q@K.swapaxes(-2,-1)/math.sqrt(Q.shape[-1])
    scores=np.where(allow,scores,-np.inf)
    shifted=scores-scores.max(-1,keepdims=True)
    weights=np.exp(shifted);weights/=weights.sum(-1,keepdims=True)
    return weights@V,weights


def threshold_for_pfa(noise_scores,alpha):
    from capstones.rf_detection.inference import threshold_for_pfa as threshold
    return threshold(noise_scores,alpha)


def operating_point(y,scores,threshold):
    from capstones.rf_detection.inference import operating_point as rates
    return rates(y,scores,threshold)
