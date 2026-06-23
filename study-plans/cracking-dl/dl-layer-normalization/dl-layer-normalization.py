import numpy as np

def layer_normalization(x, gamma, beta, eps=1e-5, mode="forward", d_output=None):
    """
    Returns: Dict with "output", "mean", "var", "x_hat", and optionally "dx", "dgamma", "dbeta".
    """
    x = np.array(x)
    
    N, D = x.shape
    mu = np.mean(x, axis=1, keepdims=True)
    sig = np.var(x, axis=1, keepdims=True)
    x_hat = (x - mu)/(np.sqrt(sig + eps))
    output = gamma * x_hat + beta
    result = {
        "output": np.round(output, 4),
        "mean": np.round(mu, 4).reshape(-1),
        "var": np.round(sig, 4).reshape(-1),
        "x_hat": np.round(x_hat, 4),
    }

    if d_output is not None:
        d_output = np.array(d_output)
        dgamma = np.sum(d_output * x_hat, axis=0)
        dbeta = np.sum(d_output, axis=0)
        dxhat = d_output * gamma
        inv_std = 1/np.sqrt(sig + eps)
        dx = (1.0/D) * inv_std * (D*dxhat - np.sum(dxhat, axis=1, keepdims=True) - x_hat * np.sum(dxhat*x_hat, axis=1, keepdims=True))
        result["dx"] = np.round(dx, 4)
        result["dgamma"] = np.round(dgamma, 4)
        result["dbeta"] = np.round(dbeta, 4)

    return result