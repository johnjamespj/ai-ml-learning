"""Independent synthetic complex-IQ recordings. No real radar or private data."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter


@dataclass(frozen=True)
class Batch:
    iq: np.ndarray
    y: np.ndarray
    snr_db: np.ndarray
    group_ids: tuple[str, ...]


def generate(n: int, seed: int, split: int, *, length: int = 128,
             snr: float | None = None, condition: str = 'in_distribution',
             noise_only: bool = False) -> Batch:
    if n < 2 or length != 128:
        raise ValueError('This reference experiment requires n >= 2 and 128-sample recordings')
    if condition not in {'in_distribution','heldout_frequency','colored_noise','clipped'}:
        raise ValueError('Unknown condition')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed),int(split)]))
    y = np.zeros(n,dtype=int) if noise_only else np.arange(n)%2
    rng.shuffle(y)
    t=np.arange(length)
    low,high=(36,52) if condition=='heldout_frequency' else (6,28)
    bins=rng.integers(low,high,size=n)
    phase=rng.uniform(-np.pi,np.pi,size=n)
    signals=np.exp(1j*(2*np.pi*bins[:,None]*t[None,:]/length+phase[:,None]))
    kinds=rng.integers(0,3,size=n)
    chirps=kinds==1
    signals[chirps] *= np.exp(1j*np.pi*6*(t[None,:]/length)**2)
    pulses=kinds==2
    for i in np.where(pulses)[0]:
        start=int(rng.integers(0,length//2))
        gate=np.zeros(length);gate[start:start+length//2]=1
        signals[i]*=gate
    signals/=np.sqrt(np.mean(np.abs(signals)**2,axis=1,keepdims=True))
    snrs=rng.uniform(-15,5,n) if snr is None else np.full(n,float(snr))
    noise=(rng.normal(size=(n,length+128))+1j*rng.normal(size=(n,length+128)))/np.sqrt(2)
    if condition=='colored_noise':
        rho=.85
        noise=lfilter([np.sqrt(1-rho*rho)],[1,-rho],noise,axis=1)
    noise=noise[:,-length:]
    iq=noise+y[:,None]*np.sqrt(10**(snrs[:,None]/10))*signals
    if condition=='clipped':
        iq=np.clip(iq.real,-1,1)+1j*np.clip(iq.imag,-1,1)
    ids=tuple(f'{seed}:{split}:{i}' for i in range(n))
    return Batch(iq.astype(np.complex64),y,snrs,ids)
