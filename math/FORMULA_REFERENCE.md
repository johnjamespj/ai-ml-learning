# Core AI/ML Formula Reference

This is a reference, not a substitute for deriving the formulas in the math notebooks.

## Linear algebra

Dot product:
[
x^Tw=\sum_i x_iw_i
]

Projection:
[
\operatorname{proj}_u(x)=\frac{x^Tu}{u^Tu}u
]

SVD:
[
X=U\Sigma V^T
]

Eigenvector:
[
Av=\lambda v
]

## Calculus and optimization

Gradient descent:
[
\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t)
]

Least-squares gradient:
[
\nabla_w\frac1n\|Xw-y\|^2=\frac2nX^T(Xw-y)
]

Newton step:
[
\theta_{t+1}=\theta_t-H^{-1}\nabla J
]

## Probability and statistics

Bayes:
[
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
]

Gaussian:
[
p(x)=\frac1{\sqrt{2\pi\sigma^2}}
e^{-(x-\mu)^2/(2\sigma^2)}
]

MLE:
[
\hat\theta=\arg\max_\theta p(D|\theta)
]

Standard error of mean:
[
SE(\bar X)=\sigma/\sqrt n
]

## Information theory

Entropy:
[
H(p)=-\sum_xp(x)\log p(x)
]

Cross-entropy:
[
H(p,q)=-\sum_xp(x)\log q(x)
]

KL:
[
D_{KL}(p\|q)=\sum_xp(x)\log\frac{p(x)}{q(x)}
]

Softmax:
[
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}
]

Softmax-cross-entropy logit gradient:
[
\frac{\partial L}{\partial z}=p-y
]

## Linear models

Normal equations:
[
X^TX\hat w=X^Ty
]

Ridge:
[
\hat w=(X^TX+\lambda I)^{-1}X^Ty
]

Logistic model:
[
P(y=1|x)=\sigma(w^Tx+b)
]

Sigmoid:
[
\sigma(z)=1/(1+e^{-z})
]

## Neural networks

Layer:
[
z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}
]

Residual block:
[
y=x+F(x)
]

Residual Jacobian:
[
\partial y/\partial x=I+J_F
]

He variance:
[
\operatorname{Var}(W_{ij})\approx2/n_{in}
]

## Attention

[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
]

[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
]

## VAE

ELBO:
[
\log p(x)
\ge
\mathbb E_q[\log p(x|z)]
-
D_{KL}(q(z|x)\|p(z))
]

Gaussian reparameterization:
[
z=\mu+\sigma\epsilon,\quad \epsilon\sim\mathcal N(0,I)
]

## Diffusion

[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
]

## Reinforcement learning

Return:
[
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}
]

Q-learning:
[
Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]
]

## Detection

Likelihood ratio:
[
\Lambda(x)=\frac{p(x|H_1)}{p(x|H_0)}
\mathop{\gtrless}_{H_0}^{H_1}\eta
]

Known-signal matched-filter statistic in white Gaussian noise:
[
T(x)=s^Tx
]

Unknown-phase quadrature statistic:
[
T(x)=I^2+Q^2
]
