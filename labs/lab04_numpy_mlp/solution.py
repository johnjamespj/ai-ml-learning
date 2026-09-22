import numpy as np

def init_params(d_in,d_hidden,d_out,seed=0):
    rng=np.random.default_rng(seed)
    return {
        "W1":rng.normal(0,np.sqrt(2/d_in),(d_in,d_hidden)),
        "b1":np.zeros(d_hidden),
        "W2":rng.normal(0,np.sqrt(2/d_hidden),(d_hidden,d_out)),
        "b2":np.zeros(d_out),
    }

def forward(X,p):
    Z1=X@p["W1"]+p["b1"]
    A1=np.maximum(0,Z1)
    logits=A1@p["W2"]+p["b2"]
    shifted=logits-logits.max(axis=1,keepdims=True)
    exp=np.exp(shifted)
    probs=exp/exp.sum(axis=1,keepdims=True)
    return probs,(Z1,A1,probs)

def cross_entropy(probs,y):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)),y],1e-12,1))))

def backward(X,y,p,cache):
    Z1,A1,probs=cache
    dlogits=probs.copy()
    dlogits[np.arange(len(y)),y]-=1
    dlogits/=len(y)
    dW2=A1.T@dlogits
    db2=dlogits.sum(axis=0)
    dA1=dlogits@p["W2"].T
    dZ1=dA1*(Z1>0)
    dW1=X.T@dZ1
    db1=dZ1.sum(axis=0)
    return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2}

def step(p,grads,lr):
    for k in p:
        p[k]-=lr*grads[k]

def predict(X,p):
    probs,_=forward(X,p)
    return probs.argmax(axis=1)
