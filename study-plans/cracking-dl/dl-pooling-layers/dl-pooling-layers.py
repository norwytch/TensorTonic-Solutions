import numpy as np

def pooling(input, pool_size, stride, pool_type):
    """
    Returns: 3D list with pooled values rounded to 4 decimal places.
    """
    input = np.array(input, dtype=np.float32)
    C, H, W = input.shape
    k = pool_size
    H_out = (H-pool_size)//stride+1
    W_out = (W-pool_size)//stride+1
    output = np.zeros((C, H_out, W_out))
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                window = input[c, i*stride:i*stride+k, j*stride:j*stride+k]
                if pool_type == 'max':
                    output[c,i,j] = np.max(window)
                if pool_type == 'average':
                    output[c,i,j] = np.mean(window)
    return output