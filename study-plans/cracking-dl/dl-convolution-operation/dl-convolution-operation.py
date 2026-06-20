import numpy as np

def conv2d(input, filters, bias=None, padding=0, stride=1):
    """
    Returns: 3D list of shape (C_out, H_out, W_out) with values rounded to 4 decimal places.
    """
    input = np.array(input, dtype=float)
    padded =  np.pad(input, ((0,0),(padding, padding),(padding,padding)))
    filters = np.array(filters, dtype=float)
    if bias is not None:
        bias = np.array(bias, dtype=float)
    C_in, H, W = padded.shape
    C_out, k_H, k_W = filters.shape[0], filters.shape[2], filters.shape[3]
    H_out = ((H-k_H)//stride)+1
    W_out = ((W-k_W)//stride)+1
    out = np.zeros((C_out, H_out, W_out))
    for c_o in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                val = 0.0
                for c in range(C_in):
                    for m in range(k_H):
                        for n in range(k_W):
                            val+=padded[c, i*stride+m, j*stride+n]*filters[c_o, c, m, n]
                if bias is not None:
                    val += bias[c_o]
                out[c_o, i, j] = val

    return np.round(out, 4).tolist()
