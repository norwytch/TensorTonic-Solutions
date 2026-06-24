import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pe = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for i in range(d_model):
            denom = 10000 ** ((2 * (i // 2)) / d_model)
            if i % 2 == 0:
                pe[pos, i] = np.sin(pos/denom) 
            else:
                pe[pos, i] = np.cos(pos/denom) 
    return pe