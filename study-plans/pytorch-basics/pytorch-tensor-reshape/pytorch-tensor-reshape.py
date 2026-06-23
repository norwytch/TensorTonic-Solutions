import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    result = None
    x = torch.tensor(x, dtype=torch.float)
    
    if op == 'flatten':
        result = torch.flatten(x)

    if op == 'squeeze':
        result = torch.squeeze(x)

    if op == 'transpose':
        result = torch.transpose(x, 0, 1)
    
    return result.tolist()
