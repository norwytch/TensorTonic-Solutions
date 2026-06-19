import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    if method == 'zeros':
        return torch.zeros((shape))
    if method == 'ones':
        return torch.ones((shape))
    if method == 'full':
        return torch.full(shape, value)
    else:
        return None
