import importlib, os
import numpy as np
m=importlib.import_module(f"labs.lab10_signal_detection.{os.getenv('LAB_TARGET','solution')}")

def test_shapes():
    X,y=m.generate_batch(20,-5,seed=1)
    assert X.shape==(20,128)
    assert y.shape==(20,)
    assert m.matched_filter_score(X).shape==(20,)

def test_high_snr_easier():
    noise,_=m.generate_batch(500,-20,signal_probability=0,seed=2)
    th=m.threshold_for_pfa(m.matched_filter_score(noise),.05)
    Xlo,ylo=m.generate_batch(1000,-15,seed=3)
    Xhi,yhi=m.generate_batch(1000,5,seed=4)
    pd_lo,_=m.detection_rate(m.matched_filter_score(Xlo),ylo,th)
    pd_hi,_=m.detection_rate(m.matched_filter_score(Xhi),yhi,th)
    assert pd_hi>pd_lo
