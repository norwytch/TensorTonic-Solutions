import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """

    pred = torch.as_tensor(pred)
    target = torch.as_tensor(target)

    if method == 'mse':
        pred = pred.float()
        target = target.float()
        loss = ((pred - target)**2).mean()
        return loss.item()
    
    if method == 'cross_entropy':
        N = pred.shape[0]
        lse = torch.logsumexp(pred, dim=1)
        correct = pred[torch.arange(N), target]
        loss = (lse - correct).mean()
        return loss.item()

    if method == 'huber':
        if delta is not None:
            a = pred-target
            loss = torch.where(torch.abs(a) <= delta, 0.5*(a**2), delta*(torch.abs(a) - .5*delta))
        return loss.mean().item()