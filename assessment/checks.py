"""Public numerical checks. Passing these is not an academic credential."""
from __future__ import annotations
import numpy as np


def check_submission(exam: str, namespace: dict) -> dict:
    def f(name):
        fn=namespace.get(name)
        if not callable(fn): raise AssertionError(f'Missing implementation: {name}')
        return fn
    rng=np.random.default_rng(8)
    if exam=='01_foundations':
        X=rng.normal(size=(11,3)); w=rng.normal(size=3); y=rng.normal(size=11)
        got=f('mse_gradient')(X,y,w)
        expected=2*X.T@(X@w-y)/len(y)
        np.testing.assert_allclose(got,expected,rtol=1e-7,atol=1e-8)
        assert np.asarray(got).shape==(3,)
        assert abs(f('posterior_positive')(.001,.9,.01)-(.0009/(.0009+.00999)))<1e-10
    elif exam=='02_classical_ml':
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X=rng.normal(size=(80,3)); y=(X[:,0]>.2).astype(int)
        model=f('build_pipeline')()
        assert isinstance(model,Pipeline), 'Return a preprocessing/estimator Pipeline'
        assert any(isinstance(x,StandardScaler) for x in model.named_steps.values())
        model.fit(X,y)
        scaler=next(x for x in model.named_steps.values() if isinstance(x,StandardScaler))
        np.testing.assert_allclose(scaler.mean_,X.mean(0))
        p=model.predict_proba(X[:4])
        assert p.shape==(4,2) and np.allclose(p.sum(1),1)
        w=rng.normal(size=3)
        expected=X.T@(1/(1+np.exp(-(X@w)))-y)/len(y)
        np.testing.assert_allclose(f('logistic_gradient')(X,y,w),expected,atol=1e-8)
    elif exam=='03_neural_networks':
        Z=rng.normal(size=(4,5)); upstream=rng.normal(size=(4,5))
        np.testing.assert_allclose(f('relu_backward')(upstream,Z),upstream*(Z>0))
        logits=np.array([[1000.,999.,998.],[-1000.,-999.,-998.]])
        y=np.array([0,2]); shifted=logits-logits.max(1,keepdims=True)
        expected=np.mean(np.log(np.exp(shifted).sum(1))-shifted[np.arange(2),y])
        got=f('cross_entropy_logits')(logits,y)
        assert np.isfinite(got) and abs(got-expected)<1e-9
    elif exam=='04_pytorch':
        import torch
        from torch import nn
        torch.manual_seed(0); torch.set_num_threads(1)
        model=nn.Linear(2,2); x=torch.tensor([[1.,0.],[-1.,0.]])
        y=torch.tensor([1,0]); opt=torch.optim.SGD(model.parameters(),lr=.1)
        # Contaminate existing gradients: the implementation must clear them.
        for p in model.parameters(): p.grad=torch.full_like(p,100.)
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        loss=f('train_step')(model,x,y,opt)
        assert np.isfinite(loss)
        assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
        assert max(p.abs().max().item() for p in model.parameters())<5, 'Clear stale gradients'
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.train(); score=f('evaluate_accuracy')(model,x,y)
        assert 0<=score<=1 and model.training, 'Restore the prior train/eval mode'
        for k,v in model.state_dict().items(): assert torch.equal(v,before[k])
    elif exam=='05_transformers':
        q=rng.normal(size=(2,4,3)); k=rng.normal(size=(2,4,3)); v=rng.normal(size=(2,4,5))
        mask=np.tril(np.ones((4,4),bool))
        out,weights=f('attention')(q,k,v,mask)
        assert out.shape==(2,4,5) and weights.shape==(2,4,4)
        assert np.all(weights[:,~mask]==0)
        np.testing.assert_allclose(weights.sum(-1),1,atol=1e-8)
        v2=v.copy(); v2[:,-1]+=100
        out2,_=f('attention')(q,k,v2,mask)
        np.testing.assert_allclose(out[:,:-1],out2[:,:-1],atol=1e-8)
        try: f('attention')(q,k,v,np.zeros((4,4),bool))
        except ValueError: pass
        else: raise AssertionError('Fully masked rows must raise ValueError, not return NaNs')
    elif exam=='06_systems':
        scores=np.arange(99,dtype=float)
        assert f('threshold_for_pfa')(scores,.05)==94.
        assert np.isinf(f('threshold_for_pfa')(np.arange(5.),.01))
        y=np.array([0,0,1,1]); s=np.array([.1,.8,.2,.9])
        r=f('operating_point')(y,s,.5)
        assert r['pd']==.5 and r['pfa']==.5 and r['tp']==1 and r['fp']==1
    else:
        raise ValueError(f'Unknown exam: {exam}')
    return {'exam':exam,'coding_checks':'passed','mastery':'not_yet_assessed',
            'next':'Human review of derivation, debugging, experimental design and oral defense is required.'}
