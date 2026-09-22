# Flagship capstone: synthetic RF detection system

**Question:** Does a more complicated learned detector improve the operating point enough to justify its cost, compared with signal-processing baselines?

This is an educational signal-presence detector, not a validated operational radar system. It uses independently generated, 128-sample complex-IQ recordings. No private or real receiver data is included.

## End-to-end implementation

Open [capstone.ipynb](capstone.ipynb). The runnable pipeline covers IQ generation, feature extraction, a Fourier matched-filter bank, energy detection, a feature logistic model, a spectrogram CNN, a temporal Transformer, probability calibration, threshold selection, independent testing, artifact export, serving, and drift diagnostics.

```bash
python -m capstones.rf_detection.train --deep --seeds 0 1 2 --output results/rf-capstone
```

For a smaller baseline-only experiment:

```bash
python -m capstones.rf_detection.train --seeds 0 --n-train 300 --n-eval 200 --output results/rf-capstone
```

## Signal and statistical assumptions

Under H0 the input is complex Gaussian noise with expected power E|n|²=1. Under H1 a normalized tone, chirp, or pulsed tone is added. SNR refers to mean signal power divided by expected noise power over the recording; it is not peak amplitude and is not calibrated dBm. Signal frequency, phase, subtype and SNR vary. The true SNR and subtype are **not** given to the learned detector as features.

The matched-filter bank computes max_k |FFT(x)_k|²/N. Its search over all Fourier bins is included when calibrating the threshold. It is a useful reference, not an assertion of optimality for arbitrary chirps, unknown colored noise, or clipped receivers.

Five data roles are separated: training learns model weights/scalers; validation selects neural checkpoints; a third split calibrates score probabilities; independent noise-only data set the decision threshold; final data estimate performance. Record IDs encode seed, split and recording index. Real data must instead group windows by original recording/session/device and avoid overlap across splits.

Thresholds use a finite-sample order statistic and strict `score > threshold`. When the calibration set cannot resolve the requested false-alarm probability, the helper returns infinity rather than inventing a precise threshold. The CLI rejects under-resolved configurations. The method relies on exchangeable calibration and deployment negatives; under domain shift, the false-alarm rate can change.

## What is measured

`metrics.csv` includes TP/FP/TN/FN, Pd and Pfa, Wilson intervals, ROC-AUC, average precision, and Brier score for each predeclared seed, model, condition and SNR. Tests include in-distribution data, unseen frequency bands, correlated noise, and clipping. One fixed threshold per fitted model is used across all these tests.

`across_seed_summary.csv` reports descriptive means and standard deviations. These are not automatically confidence intervals. `training_history.json` records validation selection. `pd_vs_snr.png` is generated from measured results, not a canned example. `latency.json` measures warm local NumPy inference and explicitly excludes HTTP/network overhead.

## Serving and serialization

The deployment artifact is the **feature logistic model from the first predeclared seed**. It is not selected using final test scores. Its scaler, coefficients, probability calibration and threshold are exported to checksummed JSON. Loading never executes a pickle. Tests compare scikit-learn, exported NumPy inference and the API.

```bash
MODEL_PATH=results/rf-capstone/model.json uvicorn capstones.rf_detection.app:app --host 127.0.0.1 --port 8000
```

Requests contain exactly 128 `[I,Q]` pairs. Inputs must be finite and bounded. Startup fails when the artifact is missing or invalid. `/health` reports readiness and the actual model hash.

```bash
docker build -f capstones/rf_detection/Dockerfile -t course-rf .
docker run --rm --read-only --tmpfs /tmp -p 127.0.0.1:8000:8000 \
  -v "$PWD/results/rf-capstone/model.json:/models/model.json:ro" course-rf
```

The container runs as a non-root user. It is a local teaching service, not an authenticated public deployment. Production work still requires authentication, request-size controls, TLS, resource budgets, monitoring and a deployment-specific safety review.

## Monitoring and rollback

`monitoring.drift_report` compares engineered feature distributions with two-sample KS diagnostics and a multiple-comparison adjustment. It does not infer accuracy from drift alone. Correlated live windows require a dependence-aware procedure. Investigate sensor configuration and provenance before retraining; do not automatically overwrite an incumbent on an unlabeled distribution change.

Retain the previous model JSON, its checksum and its evaluation report. A model version should be rolled back if schema checks fail, artifacts disagree, latency exceeds a predeclared budget, or labeled operational performance violates a requirement. Define those requirements before collecting final test evidence.

## Required learner defense

Explain the known-signal Gaussian likelihood-ratio derivation, why the bank searches frequencies, what normalization preserves, why the five splits differ, how prevalence affects probability interpretation, how false-alarm uncertainty is reported, and whether the neural alternatives actually earned their extra complexity. A negative comparison is a valid experimental result.

Extension work: real receiver capture, temporally grouped evaluation, more realistic receiver impairments, deployment of a neural candidate, and operational performance qualification remain separate projects. Never transfer synthetic performance claims directly to hardware.
