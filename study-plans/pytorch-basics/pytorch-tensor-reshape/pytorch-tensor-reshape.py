import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.tensor(x)
    if op == 'flatten':
        x = torch.flatten(x)
    if op == 'squeeze':
        x = torch.squeeze(x)
    if op == 'transpose':
        x = torch.transpose(x,0,1)
    x = x.float().tolist()
    return x