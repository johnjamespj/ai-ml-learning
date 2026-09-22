"""Numerical verification companions for eight mathematical problem sets."""
import numpy as np


def check(problem: str, fn):
    rng=np.random.default_rng(15)
    if problem=='01_matrix_derivatives':
        X=rng.normal(size=(9,4));y=rng.normal(size=9);w=rng.normal(size=4)
        got=fn(X,y,w); eps=1e-6
        expected=np.array([(np.mean((X@(w+eps*np.eye(4)[j])-y)**2)-np.mean((X@(w-eps*np.eye(4)[j])-y)**2))/(2*eps) for j in range(4)])
        np.testing.assert_allclose(got,expected,rtol=1e-5,atol=1e-7)
    elif problem=='02_likelihood_map':
        np.testing.assert_allclose(fn(14,6,2,3),(15/23))
    elif problem=='03_entropy_kl':
        p=np.array([.8,.2]);q=np.array([.5,.5])
        np.testing.assert_allclose(fn(p,q),np.sum(p*np.log(p/q)))
        assert fn(np.array([1.,0.]),np.array([.5,.5]))==np.log(2)
        assert np.isinf(fn(np.array([1.,0.]),np.array([0.,1.])))
    elif problem=='04_pca_proof':
        X=rng.normal(size=(100,3));X[:,1]=2*X[:,0]+.1*rng.normal(size=100);X-=X.mean(0)
        v=np.asarray(fn(X));_,_,V=np.linalg.svd(X,full_matrices=False)
        assert v.shape==(3,) and abs(np.linalg.norm(v)-1)<1e-8
        assert abs(v@V[0])>1-1e-8
    elif problem=='05_backprop':
        A=rng.normal(size=(6,4));D=rng.normal(size=(6,3))
        W,b=fn(A,D)
        np.testing.assert_allclose(W,A.T@D);np.testing.assert_allclose(b,D.sum(0))
    elif problem=='06_attention_scaling':
        scores=rng.normal(size=(7,5))*100
        p=fn(scores)
        assert np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(1),1)
        np.testing.assert_allclose(p,fn(scores+10000),atol=1e-10)
    elif problem=='07_bellman':
        P=rng.uniform(size=(4,2,4));P/=P.sum(-1,keepdims=True)
        R=rng.normal(size=(4,2,4));V=rng.normal(size=4);gamma=.8
        expected=(P*(R+gamma*V[None,None,:])).sum(-1).max(1)
        np.testing.assert_allclose(fn(V,P,R,gamma),expected)
        W=rng.normal(size=4)
        assert np.max(np.abs(fn(V,P,R,gamma)-fn(W,P,R,gamma)))<=gamma*np.max(np.abs(V-W))+1e-10
    elif problem=='08_detection':
        A=rng.normal(size=(5,5));C=A@A.T+np.eye(5)
        s=rng.normal(size=5);x=rng.normal(size=5)
        np.testing.assert_allclose(fn(x,s,C),s@np.linalg.solve(C,x))
    else: raise ValueError(problem)
    return {'numerical_check':'passed','proof_and_assumptions':'requires human review'}
