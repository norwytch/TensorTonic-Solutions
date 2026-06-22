import torch

def causal_attention(Q, K, V):
    """
    Returns: masked attention output tensor
    """
    batch, seq_q, d_k = Q.shape
    batch, seq_k, d_k = K.shape
    scores = (Q@K.transpose(-2,-1)/(d_k**.5))
    mask = torch.triu(torch.full((seq_q, seq_k), float('-inf')), diagonal=1)
    scores = (scores + mask)
    scores = torch.nn.functional.softmax(scores, dim=-1)
    O = scores@V
    return O