"""Run the complete educational detection comparison and export one safe API model."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from coursekit.experiments import Experiment
from .data import generate
from .inference import FEATURES,features,baseline_scores,threshold_for_pfa,operating_point,sigmoid,write_artifact,predict


def calibration(raw,y):
    scaler=StandardScaler().fit(np.asarray(raw)[:,None])
    m=LogisticRegression(C=1e3,max_iter=1000).fit(scaler.transform(np.asarray(raw)[:,None]),y)
    a=float(m.coef_[0,0]/scaler.scale_[0]);b=float(m.intercept_[0]-a*scaler.mean_[0])
    return a,b


def run(output: Path, *, seeds=(0,1,2), deep=False, n_train=1200,n_eval=800,epochs=12,pfa=.01):
    if n_eval < 100 or not 0<pfa<.5: raise ValueError('Use >=100 evaluation samples and Pfa in (0,.5)')
    if np.ceil((n_eval+1)*(1-pfa))>n_eval: raise ValueError('Threshold sample count cannot resolve requested Pfa')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config=dict(seeds=list(seeds),length=128,n_train=n_train,n_eval=n_eval,epochs=epochs,deep=deep,
                target_pfa=pfa,synthetic_prevalence=.5,training_snr_db=[-15,5],
                split_roles={'0':'train','1':'validation','2':'probability_calibration','3':'noise_threshold',
                             '100+condition*10+snr_index':'final_test'},
                limitations=['synthetic IQ only','no validated real-receiver performance','normalization and prevalence are specified, not universal'])
    (output/'config.json').write_text(json.dumps(config,indent=2))
    rows=[];histories={};test_predictions=[];summary=[];evidence=Experiment('rf-capstone',config)
    for seed in seeds:
        train=generate(n_train,seed,0);val=generate(n_eval,seed,1)
        cal=generate(n_eval,seed,2);threshold_data=generate(n_eval,seed,3,noise_only=True)
        groups=[set(b.group_ids) for b in [train,val,cal,threshold_data]]
        assert all(not groups[i]&groups[j] for i in range(4) for j in range(i))
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500)).fit(features(train.iq),train.y)
        scores={name:(lambda iq,key=name:baseline_scores(iq)[key]) for name in ('energy','matched_filter_bank')}
        scores['feature_logistic']=lambda iq:model.decision_function(features(iq))
        if deep:
            from .neural import fit
            for name in ('spectrogram_cnn','temporal_transformer'):
                score,history,_=fit(name,train,val,seed=seed,epochs=epochs)
                scores[name]=score;histories[f'{seed}:{name}']=history
        for name,score in scores.items():
            a,b=calibration(score(cal.iq),cal.y)
            threshold=threshold_for_pfa(sigmoid(a*score(threshold_data.iq)+b),pfa)
            if name=='feature_logistic':
                sc,lr=model.steps[0][1],model.steps[1][1]
                payload_model=dict(length=128,features=list(FEATURES),mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),
                    coef=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]),calibration_coef=a,calibration_intercept=b,
                    threshold=threshold,seed=seed,calibration_prevalence=.5,
                    use='Educational synthetic RF signal-presence detection; not a validated operational radar detector')
                payload=write_artifact(output/f'model_seed{seed}.json',payload_model)
                original=sigmoid(a*score(val.iq)+b)
                exported,_=predict(payload,val.iq)
                np.testing.assert_allclose(exported,original,rtol=1e-10,atol=1e-10)
                if seed==seeds[0]: write_artifact(output/'model.json',payload_model)
            for ci,condition in enumerate(('in_distribution','heldout_frequency','colored_noise','clipped')):
                for si,snr in enumerate((-20,-15,-10,-5,0,5)):
                    test=generate(n_eval,seed,100+10*ci+si,snr=snr,condition=condition)
                    assert not set(test.group_ids)&set.union(*groups)
                    probabilities=sigmoid(a*score(test.iq)+b)
                    metrics=operating_point(test.y,probabilities,threshold)
                    row=dict(seed=seed,model=name,condition=condition,snr_db=snr,threshold=threshold,
                        roc_auc=roc_auc_score(test.y,probabilities),average_precision=average_precision_score(test.y,probabilities),
                        brier=brier_score_loss(test.y,probabilities),**metrics)
                    rows.append(row)
                    if condition=='in_distribution' and snr==0:
                        evidence.log(name,seed,{'pd_at_0db':metrics['pd'],'pfa_at_0db':metrics['pfa'],'brier_at_0db':row['brier']})
        print(f'Finished seed {seed}: {len(scores)} models',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(output/'metrics.csv',index=False)
    frame.groupby(['model','condition','snr_db'])[['pd','pfa','brier']].agg(['mean','std']).to_csv(output/'across_seed_summary.csv')
    (output/'training_history.json').write_text(json.dumps(histories,indent=2))
    import matplotlib.pyplot as plt
    for name,part in frame[frame.condition=='in_distribution'].groupby('model'):
        grouped=part.groupby('snr_db').pd.agg(['mean','std'])
        plt.plot(grouped.index,grouped['mean'],marker='o',label=name)
    plt.xlabel('Injected in-window SNR (dB)');plt.ylabel('Probability of detection')
    plt.ylim(0,1);plt.legend();plt.title('Synthetic in-distribution test; fixed calibrated thresholds')
    plt.savefig(output/'pd_vs_snr.png',dpi=150,bbox_inches='tight');plt.close()
    from .inference import load_artifact
    payload=load_artifact(output/'model.json');sample=generate(2,seeds[0],999).iq[0]
    for _ in range(20): predict(payload,sample)
    durations=[]
    for _ in range(300):
        start=time.perf_counter_ns();predict(payload,sample);durations.append((time.perf_counter_ns()-start)/1e6)
    latency={'scope':'local NumPy inference, batch=1, not HTTP or network latency',
             'p50_ms':float(np.quantile(durations,.5)),'p95_ms':float(np.quantile(durations,.95)),
             'samples':len(durations),'warmups':20}
    (output/'latency.json').write_text(json.dumps(latency,indent=2))
    request={'iq':np.column_stack([sample.real,sample.imag]).tolist()}
    (output/'sample_request.json').write_text(json.dumps(request))
    evidence.finish()
    (output/'REPORT.md').write_text('# Synthetic RF detector experiment\n\n'
        f'Measured {len(frame)} model/seed/condition/SNR combinations. See metrics.csv for counts, Wilson intervals, ranking and probability scores.\n\n'
        'Thresholds were set only on the noise calibration split and held fixed across final tests.\n\n'
        'The served artifact is the feature logistic model from the first predeclared seed, not a post-test selected winner.\n\n'
        'Different seeds vary data and initialization; their standard deviation is descriptive, not a confidence interval for real-world deployment.\n\n'
        'No received hardware IQ, calibrated dBm, deployed receiver, online adaptation, or operational reliability has been validated.\n')
    return frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/rf-capstone'))
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--deep',action='store_true')
    p.add_argument('--n-train',type=int,default=1200);p.add_argument('--n-eval',type=int,default=800)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--pfa',type=float,default=.01)
    a=p.parse_args();run(a.output,seeds=tuple(a.seeds),deep=a.deep,n_train=a.n_train,n_eval=a.n_eval,epochs=a.epochs,pfa=a.pfa)
if __name__=='__main__': main()
