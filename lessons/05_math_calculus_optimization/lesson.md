# Lesson 05: Calculus, gradients and optimization

## Central idea
Training usually means choosing parameters that minimize an objective.

For mean squared error:

L = mean((y_hat - y)^2)

A derivative tells how a small parameter change changes the loss. With many parameters, those derivatives form the **gradient**.

Gradient descent:

theta_new = theta_old - learning_rate * gradient

## Numerical derivative
```python
def f(x):
    return x**2

x = 3.0
h = 1e-5
numerical = (f(x+h) - f(x-h)) / (2*h)
analytical = 2*x

print(numerical, analytical)
```

## Gradient descent
```python
x = 10.0
lr = 0.1

for _ in range(50):
    grad = 2*x
    x -= lr*grad

print(x)
```

## Chain rule
Neural networks compose functions. Backpropagation is an efficient application of the chain rule through that composition.

If y=f(g(x)), then:

dy/dx = df/dg * dg/dx

## Experiments
Repeat gradient descent with learning rates 0.001, 0.1, 0.9, 1.0 and 1.1. Explain convergence, oscillation and divergence.

## Exercises
Differentiate:
1. x^2
2. 3x^2 + 2x + 1
3. (wx+b-y)^2 with respect to w
4. the same expression with respect to b

Then connect #3 and #4 to the regression code from Lesson 01.
