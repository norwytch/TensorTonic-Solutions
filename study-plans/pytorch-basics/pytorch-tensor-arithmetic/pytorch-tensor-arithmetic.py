import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    result = None
    x = torch.tensor(x)
    y = torch.tensor(y)
    
    if op=='add':
        result = x + y

    if op=='multiply':
        result = x*y

    if op=='matmul':
        result = x@y

    if op=='power':
        result = x**y

    if op=='max':
        result = torch.max(x,y)


    return result.tolist()