import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    mu = x.mean(axis=-1, keepdims=True)
    sig = x.var(axis=-1, keepdims=True)
    numer = x-mu
    denom = np.sqrt(sig + eps)
    x_hat = numer/denom
    output = gamma * x_hat + beta
    return output