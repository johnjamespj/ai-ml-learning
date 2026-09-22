import numpy as np

def predict(X, w, b):
    # TODO
    raise NotImplementedError

def mse(y_true, y_pred):
    # TODO
    raise NotImplementedError

def gradients(X, y, w, b):
    # Return dw, db for mean squared error.
    # TODO
    raise NotImplementedError

def fit(X, y, lr=0.01, steps=2000):
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    history = []
    for _ in range(steps):
        pred = predict(X, w, b)
        history.append(mse(y, pred))
        dw, db = gradients(X, y, w, b)
        w -= lr * dw
        b -= lr * db
    return w, b, np.asarray(history)

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.uniform(-5, 5, (300, 1))
    y = 3 * X[:, 0] - 4 + rng.normal(0, 0.5, len(X))
    w, b, history = fit(X, y)
    print("w:", w, "b:", b, "final loss:", history[-1])
