import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    m = logits.max(dim = 1, keepdim=True).values
    numer = (logits - m).exp()
    denom = numer.sum(dim = 1, keepdim=True)

    return numer/denom
