"""NumPy-only inference and non-executable JSON model artifacts."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np

FEATURES=('mean_power','power_std','crest_power','spectral_peak_fraction',
          'spectral_entropy','fourth_moment_ratio','lag1_coherence','spectral_centroid')


def validate_iq(iq) -> np.ndarray:
    x=np.asarray(iq)
    if x.ndim==1: x=x[None,:]
    if x.ndim!=2 or x.shape[1]!=128 or not np.iscomplexobj(x):
        raise ValueError('Expected a complex array of shape (n,128) or (128,)')
    if not np.isfinite(x).all() or np.max(np.abs(x),initial=0)>1e6:
        raise ValueError('IQ must be finite and have magnitude <= 1e6')
    return x.astype(np.complex128,copy=False)


def features(iq) -> np.ndarray:
    x=validate_iq(iq);p=np.abs(x)**2;mean=p.mean(1);eps=1e-12
    spec=np.abs(np.fft.fft(x,axis=1))**2
    normalized=spec/(spec.sum(1,keepdims=True)+eps)
    entropy=-(normalized*np.log(normalized+eps)).sum(1)/np.log(x.shape[1])
    lag=np.abs((x[:,1:]*x[:,:-1].conj()).mean(1))/(mean+eps)
    centroid=(normalized*np.abs(np.fft.fftfreq(x.shape[1]))).sum(1)
    return np.column_stack([mean,p.std(1),p.max(1)/(mean+eps),normalized.max(1),entropy,
                            (p*p).mean(1)/(mean*mean+eps),lag,centroid])


def baseline_scores(iq) -> dict[str,np.ndarray]:
    x=validate_iq(iq)
    return {'energy':np.mean(np.abs(x)**2,axis=1),
            # Bank of unit-energy Fourier templates. Threshold calibration includes the search over bins.
            'matched_filter_bank':(np.abs(np.fft.fft(x,axis=1))**2).max(1)/x.shape[1]}


def sigmoid(z):
    z=np.asarray(z,float)
    return np.exp(-np.logaddexp(0,-z))


def threshold_for_pfa(scores, alpha: float) -> float:
    x=np.asarray(scores,float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or not 0<alpha<1:
        raise ValueError('Need finite 1-D scores and alpha in (0,1)')
    rank=math.ceil((len(x)+1)*(1-alpha))
    # Strict score > threshold; finite calibration cannot resolve arbitrarily tiny Pfa.
    return math.inf if rank>len(x) else float(np.sort(x)[rank-1])


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float,float]:
    if total<=0 or not 0<=successes<=total: raise ValueError('Invalid binomial counts')
    p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    radius=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return (0. if successes==0 else max(0.,center-radius),
            1. if successes==total else min(1.,center+radius))


def operating_point(y,scores,threshold) -> dict:
    y=np.asarray(y);s=np.asarray(scores,float)
    if y.ndim!=1 or s.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(s).all():
        raise ValueError('Aligned binary labels and finite scores are required')
    positives=int((y==1).sum());negatives=int((y==0).sum())
    if not positives or not negatives: raise ValueError('Both classes required for Pd and Pfa')
    pred=s>threshold
    tp=int((pred & (y==1)).sum());fp=int((pred & (y==0)).sum())
    dl,du=wilson(tp,positives);fl,fu=wilson(fp,negatives)
    return dict(tp=tp,fp=fp,fn=positives-tp,tn=negatives-fp,pd=tp/positives,pfa=fp/negatives,
                pd_low=dl,pd_high=du,pfa_low=fl,pfa_high=fu,n_signal=positives,n_noise=negatives)


def _canonical(value: dict) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def write_artifact(path: Path, model: dict) -> dict:
    payload={'schema_version':1,'model':model,'sha256':hashlib.sha256(_canonical(model)).hexdigest()}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,allow_nan=False))
    return payload


def load_artifact(path: str | Path) -> dict:
    path=Path(path)
    if path.stat().st_size>100_000: raise ValueError('Model JSON is unexpectedly large')
    payload=json.loads(path.read_text())
    if payload.get('schema_version')!=1: raise ValueError('Unsupported schema')
    m=payload['model']
    if payload.get('sha256')!=hashlib.sha256(_canonical(m)).hexdigest(): raise ValueError('Artifact checksum mismatch')
    if m['features']!=list(FEATURES) or m['length']!=128: raise ValueError('Feature/schema mismatch')
    for key in ('mean','scale','coef'):
        a=np.asarray(m[key],float)
        if a.shape!=(len(FEATURES),) or not np.isfinite(a).all(): raise ValueError(f'Invalid {key}')
    if np.any(np.asarray(m['scale'])<=0): raise ValueError('Scales must be positive')
    for key in ('intercept','calibration_coef','calibration_intercept','threshold'):
        if not np.isfinite(m[key]): raise ValueError(f'Invalid {key}')
    if not 0<=m['threshold']<=1: raise ValueError('Probability threshold out of range')
    return payload


def predict(payload: dict, iq) -> tuple[np.ndarray,np.ndarray]:
    m=payload['model'];F=features(iq)
    raw=((F-np.asarray(m['mean']))/np.asarray(m['scale']))@np.asarray(m['coef'])+m['intercept']
    probability=sigmoid(m['calibration_coef']*raw+m['calibration_intercept'])
    return probability, probability>m['threshold']
