import numpy as np

def generate_batch(n,snr_db,signal_probability=.5,length=128,freq=.125,seed=0):
    rng=np.random.default_rng(seed)
    y=(rng.random(n)<signal_probability).astype(int)
    phase=rng.uniform(0,2*np.pi,n)
    t=np.arange(length)
    noise=rng.normal(0,1,(n,length))
    signal=np.sin(2*np.pi*freq*t[None,:]+phase[:,None])
    signal_power=.5
    target_snr=10**(snr_db/10)
    amp=np.sqrt(target_snr/signal_power)
    X=noise+y[:,None]*amp*signal
    return X,y

def matched_filter_score(X,freq=.125):
    t=np.arange(X.shape[1])
    s=np.sin(2*np.pi*freq*t); c=np.cos(2*np.pi*freq*t)
    I=X@s; Q=X@c
    return (I**2+Q**2)/X.shape[1]

def threshold_for_pfa(noise_scores,pfa=.01):
    return float(np.quantile(noise_scores,1-pfa))

def detection_rate(scores,labels,threshold):
    pred=scores>=threshold
    pos=labels==1; neg=~pos
    pd=float(pred[pos].mean()) if pos.any() else float("nan")
    pfa=float(pred[neg].mean()) if neg.any() else float("nan")
    return pd,pfa
