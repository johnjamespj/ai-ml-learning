# Core AI/ML Formula Reference

This is a reference, not a substitute for deriving the formulas in the math notebooks. Define the vector convention and loss normalization before using a formula.

## Linear algebra

Dot product:
$$
x^Tw=\sum_i x_iw_i
$$

Projection onto a nonzero vector:
$$
\operatorname{proj}_u(x)=\frac{x^Tu}{u^Tu}u
$$

SVD:
$$
X=U\Sigma V^T
$$

Eigenvector:
$$
Av=\lambda v
$$

## Calculus and optimization

Gradient descent:
$$
\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t)
$$

Least-squares gradient, with samples as rows:
$$
\nabla_w\frac1n\|Xw-y\|^2=\frac2nX^T(Xw-y)
$$

Newton step, when the Hessian is invertible:
$$
\theta_{t+1}=\theta_t-H^{-1}\nabla J
$$

Solve the linear system rather than explicitly forming an inverse in numerical code.

## Probability and statistics

Bayes, for nonzero conditioning probability:
$$
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
$$

Gaussian density:
$$
p(x)=\frac1{\sqrt{2\pi\sigma^2}}
e^{-(x-\mu)^2/(2\sigma^2)}
$$

MLE:
$$
\hat\theta=\arg\max_\theta p(D|\theta)
$$

Standard error of the mean for independent, identically distributed observations with variance sigma squared:
$$
SE(\bar X)=\sigma/\sqrt n
$$

## Information theory

Entropy:
$$
H(p)=-\sum_xp(x)\log p(x)
$$

Cross-entropy:
$$
H(p,q)=-\sum_xp(x)\log q(x)
$$

KL divergence:
$$
D_{KL}(p\|q)=\sum_xp(x)\log\frac{p(x)}{q(x)}
$$

Use the zero-probability limit convention; positive mass under p with zero mass under q gives infinite KL. Natural logarithms give nats, while base-2 logarithms give bits.

Softmax:
$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}
$$

Softmax-cross-entropy logit gradient for one example and a target distribution summing to one:
$$
\frac{\partial L}{\partial z}=p-y
$$

A batch-mean loss adds the corresponding division by batch size.

## Linear models

Normal equations:
$$
X^TX\hat w=X^Ty
$$

Ridge for the objective squared residual norm plus lambda times squared weight norm:
$$
\hat w=(X^TX+\lambda I)^{-1}X^Ty
$$

A mean-squared objective changes lambda's scaling. An unpenalized intercept must be handled separately.

Logistic model:
$$
P(y=1|x)=\sigma(w^Tx+b)
$$

Sigmoid:
$$
\sigma(z)=1/(1+e^{-z})
$$

## Neural networks

Layer under a column-vector convention:
$$
z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}
$$

Residual block:
$$
y=x+F(x)
$$

Residual Jacobian:
$$
\partial y/\partial x=I+J_F
$$

He initialization variance for ReLU under its usual independence and symmetry approximations:
$$
\operatorname{Var}(W_{ij})\approx2/n_{in}
$$

## Attention

Using positions as rows:
$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Softmax operates across keys for each query; apply causal or padding restrictions before softmax.

## VAE

ELBO:
$$
\log p(x)
\ge
\mathbb E_q[\log p(x|z)]
-
D_{KL}(q(z|x)\|p(z))
$$

Diagonal-Gaussian reparameterization, with elementwise multiplication:
$$
z=\mu+\sigma\odot\epsilon,\quad \epsilon\sim\mathcal N(0,I)
$$

## Diffusion

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
$$

Here epsilon is standard Gaussian and alpha-bar is the cumulative product of one minus the forward noise variances.

## Reinforcement learning

Return:
$$
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}
$$

Q-learning for a nonterminal transition:
$$
Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]
$$

A genuinely terminal transition has no bootstrap term. A time-limit truncation is not automatically terminal in the underlying task.

## Detection

Likelihood ratio:
$$
\Lambda(x)=\frac{p(x|H_1)}{p(x|H_0)}
\mathop{\gtrless}_{H_0}^{H_1}\eta
$$

Known real-valued deterministic signal in white Gaussian noise:
$$
T(x)=s^Tx
$$

For complex observations use the corresponding conjugate-inner-product likelihood derivation. Colored noise requires covariance weighting.

Unknown-phase quadrature energy for equal-energy orthogonal templates:
$$
T(x)=I^2+Q^2
$$

Template normalization and any frequency search affect the null distribution. Calibrate the threshold for the actual statistic, then estimate false alarms on independent test data.
