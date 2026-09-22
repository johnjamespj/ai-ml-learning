import numpy as np

def sigmoid(z):
    # TODO: implement a numerically reasonable sigmoid
    raise NotImplementedError

def bce(y, p, eps=1e-12):
    # TODO
    raise NotImplementedError

def gradients(X, y, w, b):
    # TODO
    raise NotImplementedError

def fit(X, y, lr=0.1, steps=1500):
    w=np.zeros(X.shape[1]); b=0.0; history=[]
    for _ in range(steps):
        p=sigmoid(X@w+b)
        history.append(bce(y,p))
        dw,db=gradients(X,y,w,b)
        w-=lr*dw; b-=lr*db
    return w,b,np.asarray(history)

def predict_proba(X,w,b):
    # TODO
    raise NotImplementedError

def predict(X,w,b,threshold=.5):
    # TODO
    raise NotImplementedError
