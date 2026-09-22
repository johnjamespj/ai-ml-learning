import numpy as np

def sigmoid(z):
    z=np.asarray(z,dtype=float)
    out=np.empty_like(z)
    pos=z>=0
    out[pos]=1/(1+np.exp(-z[pos]))
    ez=np.exp(z[~pos])
    out[~pos]=ez/(1+ez)
    return out

def bce(y,p,eps=1e-12):
    p=np.clip(p,eps,1-eps)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def gradients(X,y,w,b):
    p=sigmoid(X@w+b)
    err=p-y
    return X.T@err/len(X), float(np.mean(err))

def fit(X,y,lr=.1,steps=1500):
    w=np.zeros(X.shape[1]); b=0.0; history=[]
    for _ in range(steps):
        p=sigmoid(X@w+b)
        history.append(bce(y,p))
        dw,db=gradients(X,y,w,b)
        w-=lr*dw; b-=lr*db
    return w,b,np.asarray(history)

def predict_proba(X,w,b):
    return sigmoid(X@w+b)

def predict(X,w,b,threshold=.5):
    return (predict_proba(X,w,b)>=threshold).astype(int)
