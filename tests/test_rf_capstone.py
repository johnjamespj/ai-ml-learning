import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from capstones.rf_detection.data import generate
from capstones.rf_detection.inference import (FEATURES,features,baseline_scores,threshold_for_pfa,
    operating_point,wilson,write_artifact,load_artifact,predict)
from capstones.rf_detection.app import create_app


def fitted_payload(tmp_path):
    batch=generate(200,7,0)
    F=features(batch.iq);sc=StandardScaler().fit(F)
    model=LogisticRegression(max_iter=1000).fit(sc.transform(F),batch.y)
    m=dict(length=128,features=list(FEATURES),mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),
           coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),
           calibration_coef=1.,calibration_intercept=0.,threshold=.65,seed=7)
    path=tmp_path/'model.json';payload=write_artifact(path,m)
    return path,payload,sc,model


def test_disjoint_reproducible_groups_and_noise():
    a=generate(100,2,0);b=generate(100,2,1)
    assert not set(a.group_ids)&set(b.group_ids)
    np.testing.assert_array_equal(a.iq,generate(100,2,0).iq)
    noise=generate(2000,4,3,noise_only=True)
    assert abs(np.mean(np.abs(noise.iq)**2)-1)<.02


def test_phase_invariant_features():
    x=generate(20,4,0).iq.astype(np.complex128)
    np.testing.assert_allclose(features(x),features(x*np.exp(.73j)),atol=1e-10)
    assert features(x).shape==(20,len(FEATURES))


def test_finite_threshold_rank_and_ties():
    assert threshold_for_pfa(np.arange(99.),.05)==94
    assert np.isinf(threshold_for_pfa(np.arange(5.),.01))
    assert not (np.ones(100)>threshold_for_pfa(np.ones(100),.1)).any()


def test_counts_and_finite_intervals():
    r=operating_point([0,0,1,1],[.1,.8,.2,.9],.5)
    assert (r['tp'],r['fp'],r['pd'],r['pfa'])==(1,1,.5,.5)
    low,high=wilson(0,1000)
    assert low==0 and 0<high<.01


def test_independent_high_snr_is_easier():
    low=generate(1000,0,100,snr=-20)
    high=generate(1000,0,101,snr=5)
    assert np.median(baseline_scores(high.iq)['matched_filter_bank'][high.y==1])>np.median(baseline_scores(low.iq)['matched_filter_bank'][low.y==1])


def test_json_serialization_matches_sklearn(tmp_path):
    path,payload,sc,model=fitted_payload(tmp_path)
    x=generate(40,7,1).iq
    p,_=predict(load_artifact(path),x)
    np.testing.assert_allclose(p,model.predict_proba(sc.transform(features(x)))[:,1],atol=1e-12)
    broken=json.loads(path.read_text());broken['model']['coef'][0]+=1
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='checksum'): load_artifact(path)


def test_api_and_offline_prediction_are_equivalent(tmp_path):
    path,payload,_,_=fitted_payload(tmp_path)
    x=generate(2,5,1).iq[0]
    request={'iq':np.column_stack([x.real,x.imag]).tolist()}
    with TestClient(create_app(path)) as client:
        assert client.get('/health').json()['status']=='ready'
        response=client.post('/predict',json=request)
        assert response.status_code==200
        p,label=predict(payload,x)
        assert abs(response.json()['probability']-p[0])<1e-12
        assert response.json()['signal_present']==bool(label[0])
        assert client.post('/predict',json={'iq':[[0,0]]}).status_code==422
        assert client.post('/predict',json={'iq':[[1e8,0]]*128}).status_code==422


def test_missing_artifact_fails_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path/'absent.json')): pass


def test_neural_shapes_and_gradient():
    torch=pytest.importorskip('torch')
    from capstones.rf_detection.neural import spectrograms,SpectrogramCNN,TemporalTransformer
    x=spectrograms(generate(4,0,2).iq)
    assert x.shape==(4,16,17)
    for cls in (SpectrogramCNN,TemporalTransformer):
        model=cls();z=model(x)
        assert z.shape==(4,)
        z.sum().backward()
        assert all(p.grad is not None for p in model.parameters())
