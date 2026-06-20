import numpy as np

def softmax(x):
    x = x-np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e/e.sum(axis=-1, keepdims=True)

def attention(Q, K, V, mask=None, mode='forward', d_output=None):
    d_head =Q.shape[1]
    S = Q@K.T/(np.sqrt(d_head))
    if mask is not None:
        S = np.where(np.array(mask), S, -1e9)
    W = softmax(S)
    O = W@V
    attn_dict={"output": O, "attention_weights": W}
    if d_output is not None:
        dO=d_output
        dV=W.T@dO
        dW=dO@V.T
        dS=W*(dW-np.sum(dW*W, axis=-1, keepdims=True))
        dQ=dS@K/np.sqrt(d_head)
        dK=dS.T@Q/np.sqrt(d_head)
        attn_dict['dQ'] = dQ
        attn_dict['dV'] = dV
        attn_dict['dK'] = dK
    return attn_dict
        
def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads, mask=None):
    """
    Returns: Dict with "output" and "attention_weights", rounded to 4 decimal places.
    """
    Q = np.array(Q, dtype=float)
    K = np.array(K, dtype=float)
    V = np.array(V, dtype=float)
    W_q = np.array(W_q, dtype=float)
    W_k = np.array(W_k, dtype=float)
    W_v = np.array(W_v, dtype=float)
    W_o = np.array(W_o, dtype=float)
    Q_hat = Q@W_q.T
    K_hat = K@W_k.T
    V_hat = V@W_v.T
    d_model = Q.shape[1]
    seq = Q.shape[0]
    d_head = (d_model//num_heads)
    Q_hat = Q_hat.reshape(seq, num_heads, d_head).transpose(1, 0, 2)
    K_hat = K_hat.reshape(seq, num_heads, d_head).transpose(1, 0, 2)
    V_hat = V_hat.reshape(seq, num_heads, d_head).transpose(1, 0, 2)
    outs, weights = [],[]
    for head in range(num_heads):
        attn_dict = attention(Q_hat[head], K_hat[head], V_hat[head], mask)
        out_i, W_i = attn_dict['output'], attn_dict['attention_weights']
        outs.append(out_i)
        weights.append(W_i)
    concat = np.concatenate(outs, axis=1)
    output = concat @ W_o.T
    attn_weights = np.stack(weights, axis=0)
    return {"output": np.round(output, 4), "attention_weights": np.round(attn_weights, 4)}