import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X = np.array(X, dtype=float)
    y = np.array(y)
    weights = np.zeros(X.shape[1])
    bias = 0.0

    for epoch in range(epochs):
        for xi, yi, in zip(X,y):
            z = xi @ weights + bias
            y_hat = np.where(z>=0, 1, 0)
            error = yi - y_hat
            weights += lr * error * xi
            bias += lr * error

    return weights.tolist(), float(bias)