import importlib, os
import numpy as np

m = importlib.import_module(f"labs.lab01_linear_regression.{os.getenv('LAB_TARGET','solution')}")

def test_prediction_shape():
    X=np.ones((5,2)); w=np.array([2.,3.])
    assert m.predict(X,w,1.).shape==(5,)

def test_gradients_match_finite_difference():
    X=np.array([[1.],[2.],[3.]])
    y=np.array([2.,4.,6.]); w=np.array([1.2]); b=0.3
    dw, db=m.gradients(X,y,w,b)
    eps=1e-6
    num_dw=(m.mse(y,m.predict(X,w+eps,b))-m.mse(y,m.predict(X,w-eps,b)))/(2*eps)
    num_db=(m.mse(y,m.predict(X,w,b+eps))-m.mse(y,m.predict(X,w,b-eps)))/(2*eps)
    assert np.allclose(dw[0],num_dw,rtol=1e-4,atol=1e-4)
    assert np.allclose(db,num_db,rtol=1e-4,atol=1e-4)

def test_fit_recovers_line():
    x=np.linspace(-2,2,200)[:,None]
    y=3*x[:,0]-4
    w,b,h=m.fit(x,y,lr=.02,steps=2000)
    assert abs(w[0]-3)<.05
    assert abs(b+4)<.05
    assert h[-1] < h[0]
