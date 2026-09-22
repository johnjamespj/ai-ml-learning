# Instructor and reference material

Do not make this the default student route. `reference_solutions.py` contains numerical answers to the six exams. `math_reference.py` contains implementations for the eight mathematical problem sets. The guidance below covers written reasoning.

These files are public in the same repository. Folder separation reduces accidental peeking; it is not access control. Grading tests are visible too, so use a human defense and additional counterexamples before declaring mastery. CI runs reference checks to verify the course, not to certify the learner.

## Exam 1: foundations

Let r=Xw-y. The differential of L=rᵀr/n is dL=2rᵀX dw/n. Thus the weight gradient is 2Xᵀr/n, shaped d rather than n. In the broadcasting bug, an n×1 prediction minus an n-vector broadcasts to n×n; keep both one-dimensional or both column vectors. Bayes uses the total positive probability Pd*pi+Pfa*(1-pi); at the given rare prevalence the posterior is about 0.0826, despite Pd=0.9. Simulation checks must repeat independent samples; increasing n changes standard error, not the parent distribution's standard deviation.

## Exam 2: classical ML

The Bernoulli negative log-likelihood is the mean of log(1+exp(z))-yz; differentiating gives Xᵀ(sigmoid(Xw)-y)/n. A zero-mean Gaussian prior adds a squared-weight MAP penalty, with scaling depending on whether the data loss is a sum or a mean. Preprocessing and supervised feature selection must be learned separately inside each training fold. A final test set cannot select features, models, thresholds or stopping time. Recordings, not overlapping windows, are the independent split unit. Accuracy, ranking, calibrated probabilities and threshold-specific error counts answer different questions.

## Exam 3: neural networks

With row-batched data, dW2=A1ᵀdZ2 and db2=sum_rows(dZ2); dZ1=(dZ2 W2ᵀ) elementwise times the activation derivative; dW1=XᵀdZ1. For mean softmax cross-entropy, dZ2=(p-one_hot(y))/n. Divide by n once. A linear hidden layer yields XW1W2 plus one combined bias. Identical hidden initialization preserves symmetry. ReLU finite-difference checks should avoid sign changes. Stable log-sum-exp permits large losses for confidently wrong predictions without overflowing exponentials.

## Exam 4: PyTorch

For one spatial axis, output size is floor((input+2p-dilation*(kernel-1)-1)/stride+1). A convolution with bias has out_channels*(in_channels/groups*kh*kw+1) parameters. `eval()` changes dropout/normalization behavior; it does not turn off autograd. `no_grad()` controls graph tracking, not train/eval state. Training must clear stale accumulated gradients. A resumable checkpoint includes optimizer state, scheduler/scaler state when used, epoch/step, data-order/RNG states and configuration; saving only weights is inference restoration, not guaranteed training continuation.

## Exam 5: Transformers

Independent, zero-mean, unit-variance products yield a sum variance of dk. Covariance and non-unit variances change the calculation. The scale factor stabilizes the magnitude distribution under those assumptions; it is not a universal cure. Masked entries receive exactly zero probability, but a fully masked row has no categorical distribution and should be rejected. With no positional information, self-attention is permutation-equivariant; invariant pooling loses order information. A method experiment needs a baseline suited to class balance and validation separate from final testing. Retrieval labels must not be defined by the retriever's own outputs.

## Exam 6: system engineering

For Gaussian covariance C, the log likelihood ratio is sᵀC⁻¹x - 0.5*sᵀC⁻¹s for a known signal, apart from prior/threshold terms. Whitening via C=LLᵀ gives (L⁻¹s)ᵀ(L⁻¹x). The fixed threshold is calibrated without final test labels. A finite-sample order-statistic threshold assumes exchangeability and uses strict inequality to handle ties conservatively; domain shift invalidates that calibration guarantee. Zero false alarms in n negatives still has a positive confidence upper bound. Serving parity must compare identical input bytes, normalization, model version and threshold, not just matching labels on one easy example.

## Mathematical problem-set guidance

Matrix derivatives should derive the Hessian 2XᵀX/n and positive semidefiniteness from vᵀXᵀXv=||Xv||². The Beta posterior interior mode needs both posterior parameters greater than one; boundary cases are not covered by the interior formula. KL nonnegativity is proved through the log inequality/Jensen, with zero-probability cases treated separately. PCA's constrained Lagrangian gives Sv=lambda*v; maximum projected variance corresponds to the largest eigenvalue. SVD component signs are not identifiable.

Backprop is a vector-Jacobian product computation, not a separate optimization method. In a ReLU initialization calculation, distinguish the second moment from centered variance; the familiar factor of two is motivated by moment propagation. Attention scaling exercises must expose their independence assumptions. Bellman contraction uses stochastic transition rows and gamma<1; nonlinear function approximation does not preserve all tabular guarantees. Colored-noise matched filtering should use a solve or whitening, not a numerically fragile explicit inverse. Threshold intervals and uncertainty should be tied to the actual denominator and data independence assumptions.
