import torch
import torch.nn as nn

class CustomSGD(torch.optim.Optimizer):
    """
    Returns: loss or None from step()
    """

    def __init__(self, params, lr=0.01, momentum=0.0):
        defaults = {'lr': lr, 'momentum': momentum}
        super().__init__(params, defaults)
            

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                with torch.no_grad():
                    if p.grad is None:
                        continue
                    grad = p.grad
                    lr = group['lr']
                    momentum = group['momentum']
                    if 'v' not in self.state[p]:
                        self.state[p]['v'] = torch.zeros_like(p)
                    v = self.state[p]['v']
                    v = momentum * v + grad
                    self.state[p]['v'] = v
                    p -= lr * v
        return loss