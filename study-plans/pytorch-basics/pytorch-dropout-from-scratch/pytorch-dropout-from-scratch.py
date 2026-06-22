import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Returns: tensor with dropout applied
        """
        if self.training:
            randtens = torch.rand_like(x)
            mask = randtens >= self.p
            x = torch.where(mask, x/(1-self.p), 0.0)
            return x
        else:
            return x
