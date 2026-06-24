import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def single_head_attention(Q, K, V):

    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V
    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """

    Q_hat = Q@W_q
    K_hat = K@W_k
    V_hat = V@W_v
    h = num_heads
    batch, seq_len = Q_hat.shape[0], Q_hat.shape[1]
    d_model = W_q.shape[0]
    d_k = d_model//num_heads
    Q_hat = Q_hat.reshape(batch, seq_len, h, d_k)
    Q_hat = np.transpose(Q_hat, (2, 0, 1, 3))
    K_hat = K_hat.reshape(batch, seq_len, h, d_k)
    K_hat = np.transpose(K_hat, (2, 0, 1, 3))
    V_hat = V_hat.reshape(batch, seq_len, h, d_k)
    V_hat = np.transpose(V_hat, (2, 0, 1, 3))
    outputs = []
    for head in range(num_heads):
        single_output = single_head_attention(Q_hat[head], K_hat[head], V_hat[head])
        outputs.append(single_output)
    concat = np.concatenate(outputs, axis=-1)
    output = concat @ W_o
    return output