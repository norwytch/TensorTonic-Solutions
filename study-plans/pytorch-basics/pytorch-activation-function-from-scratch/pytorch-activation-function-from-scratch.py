import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x)
    result = None
    
    if method == 'relu':
        zeros = torch.zeros(x.shape)
        result = torch.max(zeros,x)

    if method == 'sigmoid':
        denom = 1 + torch.exp(-x)
        result = 1/denom

    if method == 'tanh':
        numer = torch.exp(x) - torch.exp(-x)
        denom = torch.exp(x) + torch.exp(-x)
        result = numer/denom

    if method == 'leaky_relu':
        x = torch.where(x>0, x, .01*x)
        result = x

    return result.tolist()