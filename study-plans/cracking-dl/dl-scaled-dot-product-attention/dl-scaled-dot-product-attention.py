import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None, mode="forward", d_output=None):
    """
    Returns: Dict with "output", "attention_weights", and optionally "dQ", "dK", "dV".
    """
    Q, K, V = np.array(Q, dtype=float), np.array(K, dtype=float), np.array(V, dtype=float)    
    d_k = Q.shape[1]
    S=(Q@K.T/np.sqrt(d_k))
    if mask is not None:
        S = np.where(np.array(mask), S, -1e9)
    W = softmax(S)
    O = W@V
    attn_dict = {"output": np.round(O, 4), "attention_weights": np.round(W, 4)}
    if d_output is not None:
        dO=d_output
        dV = W.T @ dO
        dW = dO @ V.T
        dS = W * (dW - np.sum(dW * W, axis=-1, keepdims=True))
        dQ = dS @ K / np.sqrt(d_k)
        dK = dS.T @ Q / np.sqrt(d_k)
        attn_dict["dQ"] = np.round(dQ, 4)
        attn_dict["dK"] = np.round(dK, 4)
        attn_dict["dV"] = np.round(dV, 4)
    return attn_dict