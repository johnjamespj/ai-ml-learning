import importlib, os
import numpy as np
m=importlib.import_module(f"labs.lab04_numpy_mlp.{os.getenv('LAB_TARGET','solution')}")

def test_probabilities():
    X=np.array([[0.,1.],[1.,0.]])
    p=m.init_params(2,4,2,0)
    probs,cache=m.forward(X,p)
    assert probs.shape==(2,2)
    assert np.allclose(probs.sum(axis=1),1)

def test_gradient_shapes():
    X=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y=np.array([0,1,1,0])
    p=m.init_params(2,5,2,0)
    probs,cache=m.forward(X,p)
    g=m.backward(X,y,p,cache)
    for k in p:
        assert g[k].shape==p[k].shape

def test_one_parameter_gradient():
    X=np.array([[.2,-.3],[.4,.7]])
    y=np.array([0,1])
    p=m.init_params(2,3,2,3)
    probs,cache=m.forward(X,p)
    g=m.backward(X,y,p,cache)
    eps=1e-6
    orig=p["W2"][0,0]
    p["W2"][0,0]=orig+eps
    lp=m.cross_entropy(m.forward(X,p)[0],y)
    p["W2"][0,0]=orig-eps
    lm=m.cross_entropy(m.forward(X,p)[0],y)
    p["W2"][0,0]=orig
    num=(lp-lm)/(2*eps)
    assert np.isclose(num,g["W2"][0,0],rtol=1e-4,atol=1e-4)
