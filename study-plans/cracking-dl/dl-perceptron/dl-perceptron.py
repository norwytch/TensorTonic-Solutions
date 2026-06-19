import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    weights = np.zeros(X.shape[1])
    bias = 0.0
    for i in range(epochs):
        for j in range(len(y)):                 
            z = X[j] @ weights + bias           
            y_hat = 1 if z >= 0 else 0
            error = y[j] - y_hat
            weights += lr * error * X[j]         
            bias    += lr * error
    
    return weights.tolist(), float(bias)