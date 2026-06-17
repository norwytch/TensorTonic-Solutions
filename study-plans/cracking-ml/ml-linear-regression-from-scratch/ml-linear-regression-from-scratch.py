import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.zeros(X.shape[1])
    bias = 0.0
    N = X.shape[0]

    for i in range(epochs):
        y_hat = X@weights + bias
        dw = (2/N) * X.T @ (y_hat - y)
        db = (2/N) * np.sum(y_hat - y)
        weights -= lr * dw
        bias -= lr * db

    return (weights, bias)