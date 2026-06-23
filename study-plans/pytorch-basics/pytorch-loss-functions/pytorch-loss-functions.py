import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    pred = torch.tensor(pred)
    target = torch.tensor(target)
    loss = None
    N = pred.shape[0]
    
    if method == 'mse':
        pred = pred.float()
        target = target.float()
        sqdiff = (target-pred)**2
        loss = sqdiff.mean()

    elif method == 'cross_entropy':
        lse = torch.logsumexp(pred, dim=1)
        correct = pred[torch.arange(N), target]
        loss = (lse - correct).mean()

    elif method == 'huber':
        if delta is not None:
            a = pred - target
            loss = torch.where(torch.abs(a) <= delta, .5*(a**2), delta*(torch.abs(a) - .5*delta))
        return loss.mean().item()
            
    return loss.item()
    
