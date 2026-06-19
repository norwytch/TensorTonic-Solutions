import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x=torch.tensor(x, dtype=torch.float)
    if method =="relu":
        zeros=torch.zeros(x.shape)
        x=torch.max(x,zeros)
    elif method=="sigmoid":
        denominator = 1 + torch.exp(-x)
        x=1/denominator
    elif method=="tanh":
        numerator = torch.exp(x) - torch.exp(-x)
        denominator = torch.exp(x) + torch.exp(-x)
        x=torch.divide(numerator, denominator)
    elif method=="leaky_relu":
        x=torch.where(x<=0, x*.01, x)

    return x.tolist()