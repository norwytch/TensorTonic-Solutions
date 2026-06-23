import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float)
    a = None
    
    if method=='relu':
        zeros = torch.zeros(x.shape)
        a = torch.max(zeros,x)

    elif method=='sigmoid':
        a = (1/(1+torch.exp(-x)))

    elif method=='tanh':
        numer = torch.exp(x) - torch.exp(-x)
        denom = torch.exp(x) + torch.exp(-x)
        a = torch.divide(numer,denom)

    elif method=='leaky_relu':
        x = torch.where(x<=0, .01*x, x)
        a = x

    return a.tolist()