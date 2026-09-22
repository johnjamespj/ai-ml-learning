"""Task specifications for the six cumulative assessments."""
EXAMS = {
'01_foundations': {
 'title':'Mathematics and scientific Python', 'prerequisites':'Lessons 00–08; Math 00–07',
 'derive':'Derive the least-squares gradient using differentials. Show every shape. Derive Bayes posterior odds and explain the base-rate effect for prevalence 0.001, Pd 0.9 and Pfa 0.01.',
 'code':'Implement mse_gradient(X,y,w) and posterior_positive(prevalence,pd,pfa). Validate probability arguments and give a clear failure for a zero-probability conditioning event.',
 'debug':'Explain and repair: X has shape (20,3), w has shape (3,1), y has shape (20,), and error = X @ w - y. Why does broadcasting produce a (20,20) matrix rather than a residual vector?',
 'design':'Design a Monte Carlo experiment verifying the sample-mean standard error decreases as 1/sqrt(n). Specify distribution, sample sizes, repeat count, seeds, plotting scale and expected discrepancy at finite sample sizes.',
 'defend':'Explain why a numerical derivative is evidence for a derivation, not a proof. Distinguish variance, standard deviation and standard error.',
 'starter': 'def mse_gradient(X, y, w):\n    raise NotImplementedError("Implement the analytical gradient")\n\ndef posterior_positive(prevalence, pd, pfa):\n    raise NotImplementedError("Implement Bayes rule with validation")\n'},
'02_classical_ml': {
 'title':'Classical ML and defensible evaluation','prerequisites':'Lessons 08–18; classical benchmark',
 'derive':'Derive the binary logistic negative log-likelihood and its weight gradient. Connect L2 regularization to both a Gaussian MAP prior and conditioning of a linear problem.',
 'code':'Implement build_pipeline() returning StandardScaler and LogisticRegression in one sklearn Pipeline, and logistic_gradient(X,y,w) for mean binary cross-entropy without an intercept.',
 'debug':'A researcher standardizes the whole dataset, selects ten features using all labels, then cross-validates. Identify both leaks and write a corrected fit/transform/select order within each training fold.',
 'design':'You have 40 recordings containing thousands of overlapping windows. Specify grouped train/validation/test splitting, a dummy baseline, nested or budgeted CV, model selection and a rare-event metric. Do not split neighboring windows randomly.',
 'defend':'Explain why accuracy can be 99% for a useless detector; distinguish ranking, calibration and an operating threshold.',
 'starter': 'def build_pipeline():\n    raise NotImplementedError("Return a preprocessing and classification pipeline")\n\ndef logistic_gradient(X, y, w):\n    raise NotImplementedError("Return a vector shaped like w")\n'},
'03_neural_networks': {
 'title':'Neural networks from first principles','prerequisites':'Lessons 19–24; NumPy MLP; backprop paper',
 'derive':'For Z1=XW1+b1, A1=ReLU(Z1), Z2=A1W2+b2, derive all cross-entropy gradients using row-batched convention. Show why two linear layers collapse into one affine map.',
 'code':'Implement relu_backward(upstream,z) and a stable mean cross_entropy_logits(logits,y) using log-sum-exp. Do not clip probabilities to hide overflow or underflow.',
 'debug':'A network reaches 50% on XOR. Its developer removed the hidden activation, divides the gradient by batch size twice, and initializes every hidden unit identically. Explain the distinct effect of each bug and how to isolate it.',
 'design':'Plan a gradient check away from ReLU kinks and a tiny-batch overfit test. Give absolute/relative tolerances, finite-difference step sizes, and an ablation with and without nonlinearity.',
 'defend':'Distinguish backpropagation from the optimizer. Explain vanishing gradients, residual Jacobians and the assumptions behind He initialization.',
 'starter': 'def relu_backward(upstream, z):\n    raise NotImplementedError("Apply the local Jacobian")\n\ndef cross_entropy_logits(logits, y):\n    raise NotImplementedError("Use stable log-sum-exp")\n'},
'04_pytorch': {
 'title':'PyTorch, training systems and computer vision','prerequisites':'Lessons 25–34; PyTorch classifier and CNN project',
 'derive':'Derive the output spatial size and parameter count of a 2-D convolution with stride, padding and dilation. Contrast this with a dense layer over all pixels.',
 'code':'Implement train_step(model,X,y,optimizer) using CrossEntropyLoss; return a scalar loss. Implement evaluate_accuracy(model,X,y) without changing weights or leaving the model in a different mode.',
 'debug':'Find and repair: softmax before CrossEntropyLoss, omitted zero_grad, CPU labels with a CUDA model, and dropout left on during validation. Explain why each can fail differently.',
 'design':'Design a controlled CNN-versus-MLP experiment with train/validation/test sets, matched data budget, justified parameter-count comparison, seed repeats, checkpoints and validation-only early stopping.',
 'defend':'Distinguish model.eval(), no_grad() and requires_grad. Explain which RNG/optimizer/model states are required for a true training resume.',
 'starter': 'def train_step(model, X, y, optimizer):\n    raise NotImplementedError("One complete optimization step")\n\ndef evaluate_accuracy(model, X, y):\n    raise NotImplementedError("Evaluate and restore the previous mode")\n'},
'05_transformers': {
 'title':'Attention, Transformers and LLM systems','prerequisites':'Lessons 35–45; Math 11; Transformer and RAG papers',
 'derive':'Derive the variance of q dot k under independent zero-mean unit-variance components. State when dividing by sqrt(dk) does not normalize variance. Explain permutation equivariance and positional information.',
 'code':'Implement attention(Q,K,V,allow_mask) using NumPy for batch/time/features tensors. Return output and weights. True means allowed. Reject a fully masked row. Do not use a high-level attention function.',
 'debug':'A decoder can see future tokens, a retrieval benchmark includes its evaluation queries in the indexed answer corpus, and generation quality is scored by fluency only. Propose three independent tests that expose these problems.',
 'design':'Specify a no-position versus position ablation with identical initialization, balanced labels, majority baseline, validation-only selection and multiple seeds. For RAG define relevance labels and Recall@k/MRR separately from answer groundedness.',
 'defend':'Explain what BERT, causal language modeling, LoRA and retrieval change, and what they leave unchanged. Distinguish a toy mechanism demo from original-benchmark reproduction.',
 'starter': 'def attention(Q, K, V, allow_mask):\n    raise NotImplementedError("Scaled scores, stable softmax and value aggregation")\n'},
'06_systems': {
 'title':'Full-system ML engineering and RF detection','prerequisites':'Lessons 52–64 and 73–77; capstone baseline',
 'derive':'Derive a known-signal likelihood-ratio test in Gaussian noise and its colored-noise whitening form. Derive precision from prevalence, Pd and Pfa. Explain why zero observed false alarms does not establish zero true Pfa.',
 'code':'Implement threshold_for_pfa(noise_scores,alpha) using sorted ascending scores and rank ceil((n+1)*(1-alpha)); return infinity if the rank exceeds n. Use strict score > threshold. Implement operating_point(y,scores,threshold) returning tp,fp,pd,pfa.',
 'debug':'A detector tunes its threshold on test negatives, reports per-window random splits from one recording, normalizes each window so signal energy disappears, and serves with a different scaler. Correct the pipeline and artifact format.',
 'design':'Specify disjoint training, validation, calibration, threshold-setting and final test data. Require Pd versus SNR, a fresh false-alarm estimate with intervals, unseen frequencies, clipping/colored-noise stress tests, serialization parity, latency percentiles and rollback criteria.',
 'defend':'State intended and prohibited use, synthetic-to-real limitations, budget, drift versus performance monitoring, data provenance and the evidence needed before deploying on an actual receiver.',
 'starter': 'def threshold_for_pfa(noise_scores, alpha):\n    raise NotImplementedError("Use the finite-sample calibration rank")\n\ndef operating_point(y, scores, threshold):\n    raise NotImplementedError("Count outcomes; use strict >")\n'}
}
