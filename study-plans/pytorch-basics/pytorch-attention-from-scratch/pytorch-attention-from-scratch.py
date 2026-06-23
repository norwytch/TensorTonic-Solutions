import torch

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: attention output tensor
    """

    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1)/(d_k**.5)
    W = torch.nn.functional.softmax(scores, dim=-1)
    output = W @ V

    return output
