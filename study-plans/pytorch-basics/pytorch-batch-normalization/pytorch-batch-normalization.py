import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    N, D = X.shape
    mu = X.mean(dim=0)
    sigma = X.var(dim=0, unbiased=False)
    X_hat = (X - mu)/torch.sqrt(sigma + eps)
    Y = gamma * X_hat + beta
    return Y