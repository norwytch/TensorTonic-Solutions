import numpy as np

def softmax(logits):
    """Stable softmax over the last axis."""
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)

def multihead(q_input, kv_input, W_q, W_k, W_v, W_o, num_heads, mask=None):
    """Multi-head attention. Q from q_input, K/V from kv_input. Returns (seq_q, d_model)."""
    seq_q, d_model = q_input.shape
    seq_k = kv_input.shape[0]
    h = num_heads
    d_head = d_model // h

    Q = q_input @ W_q.T
    K = kv_input @ W_k.T
    V = kv_input @ W_v.T

    Q = Q.reshape(seq_q, h, d_head).transpose(1, 0, 2)   
    K = K.reshape(seq_k, h, d_head).transpose(1, 0, 2)   
    V = V.reshape(seq_k, h, d_head).transpose(1, 0, 2)

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head) 
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    weights = softmax(scores)
    head_out = weights @ V                                

    head_out = head_out.transpose(1, 0, 2).reshape(seq_q, d_model)

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

    attn_output = multihead(x, x, W_q, W_k, W_v, W_o, h)
    norm1 = layer_norm(x+attn_output, gamma1, beta1)
    ffn_output = relu(norm1@W1.T+b1)@W2.T + b2
    output = layer_norm(norm1 + ffn_output, gamma2, beta2)
    
    attn_output = np.round(attn_output, 4)
    norm1 = np.round(norm1, 4)
    ffn_output = np.round(ffn_output, 4)
    output = np.round(output, 4)

    return_dict = {'attention_output': attn_output, 'norm1': norm1, 'ffn_output': ffn_output, 'output': output}

    return return_dict


def decoder_block(x, enc_output,
                  W_q1, W_k1, W_v1, W_o1,       
                  W_q2, W_k2, W_v2, W_o2,       
                  num_heads,
                  W1, b1, W2, b2,              
                  gamma1, beta1, gamma2, beta2, gamma3, beta3,
                  mask=None):
    """One transformer decoder block. mask is the causal self-attention mask."""
    x = np.array(x)
    enc_output = np.array(enc_output)
    W_q1, W_k1, W_v1, W_o1 = map(np.array, (W_q1, W_k1, W_v1, W_o1))
    W_q2, W_k2, W_v2, W_o2 = map(np.array, (W_q2, W_k2, W_v2, W_o2))
    W1, b1, W2, b2 = map(np.array, (W1, b1, W2, b2))
    gamma1, beta1, gamma2, beta2, gamma3, beta3 = map(
        np.array, (gamma1, beta1, gamma2, beta2, gamma3, beta3))
    if mask is not None:
        mask = np.array(mask)
    self_attn = multihead(x, x, W_q1, W_k1, W_v1, W_o1, num_heads, mask)
    norm1 = layer_norm(x + self_attn, gamma1, beta1)
    cross_attn = multihead(norm1, enc_output, W_q2, W_k2, W_v2, W_o2, num_heads, None)
    norm2 = layer_norm(norm1 + cross_attn, gamma2, beta2)
    ffn_output = relu(norm2 @ W1.T + b1) @ W2.T + b2
    output = layer_norm(norm2 + ffn_output, gamma3, beta3)

    return {
        'self_attn': np.round(self_attn, 4).tolist(),
        'norm1': np.round(norm1, 4).tolist(),
        'cross_attn': np.round(cross_attn, 4).tolist(),
        'norm2': np.round(norm2, 4).tolist(),
        'ffn_output': np.round(ffn_output, 4).tolist(),
        'output': np.round(output, 4).tolist(),
    }
