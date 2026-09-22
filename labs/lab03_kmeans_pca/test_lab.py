import importlib, os
import numpy as np
m=importlib.import_module(f"labs.lab03_kmeans_pca.{os.getenv('LAB_TARGET','solution')}")

def test_assignment():
    X=np.array([[0.,0.],[9.,9.]])
    C=np.array([[0.,1.],[10.,10.]])
    assert np.array_equal(m.assign_clusters(X,C),np.array([0,1]))

def test_pca_reconstruction():
    rng=np.random.default_rng(0)
    x=rng.normal(size=200)
    X=np.c_[x,2*x]
    Z,C,mean,var=m.pca_fit_transform(X,1)
    rec=m.pca_inverse(Z,C,mean)
    assert np.mean((X-rec)**2)<1e-10
