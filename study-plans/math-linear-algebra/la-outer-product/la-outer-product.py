import numpy as np

def outer_product(u, v):
    """
    Returns: float64 matrix of shape (m, n), the outer product u v^T.
    """
    u_array, v_array = np.array(u), np.array(v)
    m, n = u_array.shape[0], v_array.shape[0]
    M = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            M[i,j] = u_array[i] * v_array[j]
    return M
