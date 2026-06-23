import torch

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx
    """
    tensor = torch.tensor(values, dtype=torch.float, requires_grad=True)
    y = torch.sum(tensor**3 + 2*tensor)
    y.backward()
    return tensor.grad.tolist()
