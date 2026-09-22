"""Illustrative feature-distribution monitoring; drift is not measured accuracy."""
import numpy as np
from scipy.stats import ks_2samp
from .inference import FEATURES,features

def drift_report(reference_iq,live_iq,alpha=.01):
    old,new=features(reference_iq),features(live_iq)
    rows=[]
    for i,name in enumerate(FEATURES):
        result=ks_2samp(old[:,i],new[:,i])
        rows.append({'feature':name,'ks_statistic':float(result.statistic),'p_value':float(result.pvalue),
                     'bonferroni_flag':bool(result.pvalue<alpha/len(FEATURES))})
    return {'features':rows,'scope':'IID two-sample synthetic diagnostic; overlapping real windows require dependence-aware analysis',
            'performance':'unknown until fresh labels are available','action':'Investigate provenance and receiver conditions before retraining'}
