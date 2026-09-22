import importlib, os
import numpy as np
m=importlib.import_module(f"labs.lab02_logistic_regression.{os.getenv('LAB_TARGET','solution')}")

def test_sigmoid():
    assert np.isclose(m.sigmoid(np.array([0.]))[0],.5)
    assert m.sigmoid(np.array([100.]))[0]>.999

def test_loss_prefers_correct_probabilities():
    y=np.array([0,1])
    assert m.bce(y,np.array([.01,.99])) < m.bce(y,np.array([.4,.6]))

def test_learning():
    rng=np.random.default_rng(0)
    X=np.r_[rng.normal(-1,.4,(100,2)),rng.normal(1,.4,(100,2))]
    y=np.r_[np.zeros(100),np.ones(100)]
    w,b,h=m.fit(X,y,lr=.2,steps=500)
    pred=m.predict(X,w,b)
    assert (pred==y).mean()>.95
    assert h[-1]<h[0]
