import numpy as np

def generate_batch(n, snr_db, signal_probability=.5, length=128, freq=0.125, seed=0):
    # TODO: random phase, AWGN, labels in {0,1}
    raise NotImplementedError

def matched_filter_score(X, freq=0.125):
    # TODO: quadrature projection magnitude/energy
    raise NotImplementedError

def threshold_for_pfa(noise_scores, pfa=.01):
    # TODO
    raise NotImplementedError

def detection_rate(scores, labels, threshold):
    # TODO: return Pd and Pfa
    raise NotImplementedError
