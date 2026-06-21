import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: attention output tensor
    """
    d_k = Q.shape[-1]
    S = Q@K.transpose(-2, -1)/d_k**.5
    W = F.softmax(S, dim=-1)
    O = W @ V
    
    return O