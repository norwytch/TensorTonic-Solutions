import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    positions = np.arange(seq_len)[:,None]
    frequencies = np.arange(int(np.ceil(d_model / 2)))
    denominator = np.power(base, 2*frequencies/d_model)
    denominator = denominator[None, :]
    angles = positions/denominator
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])
        
    return pe
    # Write code here
    # result is a matrix
    # pe fills in each position in the matrix
    #d_model is both rows and columns
    #index through d_model and seq_len