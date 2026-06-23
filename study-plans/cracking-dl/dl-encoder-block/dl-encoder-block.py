import numpy as np

def softmax(logits):
    """Stable softmax over the last axis."""
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)

def multihead(x, W_q, W_k, W_v, W_o, num_heads):
    """Multi-head self-attention with Q=K=V=x. Returns (seq, d_model)."""
    seq, d_model = x.shape
    h = num_heads
    d_head = d_model // h

    # project
    Q = x @ W_q.T
    K = x @ W_k.T
    V = x @ W_v.T

    # split into heads: (seq, d_model) -> (h, seq, d_head)
    Q = Q.reshape(seq, h, d_head).transpose(1, 0, 2)
    K = K.reshape(seq, h, d_head).transpose(1, 0, 2)
    V = V.reshape(seq, h, d_head).transpose(1, 0, 2)

    # scaled dot-product attention per head (h rides as batch axis)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head)   # (h, seq, seq)
    weights = softmax(scores)
    head_out = weights @ V                                 # (h, seq, d_head)

    # merge heads: (h, seq, d_head) -> (seq, d_model)
    head_out = head_out.transpose(1, 0, 2).reshape(seq, d_model)

    # output projection
    return head_out @ W_o.T
def layer_norm(sum, gamma, beta):
    mu = np.mean(sum, axis=1, keepdims=True)
    sig = np.var(sum, axis=1, keepdims=True)
    numer = (sum - mu)
    denom = np.sqrt(sig+1e-5)
    return (numer/denom)*gamma + beta

def relu(x):
    output = np.maximum(0,x)
    return output

def encoder_block(x, W_q, W_k, W_v, W_o, num_heads, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2):
    """
    Implements one transformer encoder block.
    Returns: Dict with "attention_output", "norm1", "ffn_output", "output",
             all as list-of-lists rounded to 4 decimals.
    """
    x = np.array(x)
    seq, d_model = x.shape
    h = num_heads
    d_head = d_model//h
    
    W_q = np.array(W_q)
    W_k = np.array(W_k)
    W_v = np.array(W_v)
    W_o = np.array(W_o)
    W1 = np.array(W1)
    b1 = np.array(b1)
    W2 = np.array(W2)
    b2 = np.array(b2)
    gamma1 = np.array(gamma1)
    beta1 = np.array(beta1)
    gamma2 = np.array(gamma2)
    beta2 = np.array(beta2)

    attn_output = multihead(x, W_q, W_k, W_v, W_o, h)
    norm1 = layer_norm(x+attn_output, gamma1, beta1)
    ffn_output = relu(norm1@W1.T+b1)@W2.T + b2
    output = layer_norm(norm1 + ffn_output, gamma2, beta2)
    
    attn_output = np.round(attn_output, 4)
    norm1 = np.round(norm1, 4)
    ffn_output = np.round(ffn_output, 4)
    output = np.round(output, 4)

    return_dict = {'attention_output': attn_output, 'norm1': norm1, 'ffn_output': ffn_output, 'output': output}

    return return_dict
