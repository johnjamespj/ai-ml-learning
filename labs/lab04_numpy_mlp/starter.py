import numpy as np

def init_params(d_in, d_hidden, d_out, seed=0):
    # TODO
    raise NotImplementedError

def forward(X, p):
    # TODO: return probs and cache
    raise NotImplementedError

def cross_entropy(probs, y):
    # TODO
    raise NotImplementedError

def backward(X, y, p, cache):
    # TODO: return gradients with matching parameter keys
    raise NotImplementedError

def step(p, grads, lr):
    for k in p:
        p[k] -= lr * grads[k]

def predict(X,p):
    probs,_=forward(X,p)
    return probs.argmax(axis=1)
