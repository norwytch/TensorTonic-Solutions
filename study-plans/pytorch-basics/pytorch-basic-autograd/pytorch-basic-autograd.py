import torch

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx
    """
    values = torch.tensor(values, dtype=torch.float, requires_grad=True)
    y = torch.sum(values**3 + 2*values)
    y.backward()
    return values.grad.tolist()
