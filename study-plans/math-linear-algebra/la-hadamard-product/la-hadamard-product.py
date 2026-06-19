import numpy as np

def hadamard_product(A, B):
    """
    Returns: ndarray, the element-wise product A * B.
    """
    A, B = np.array(A), np.array(B)
    m, n = A.shape[0], A.shape[1]
    hadamard = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            hadamard[i,j] = A[i,j] * B[i,j]

    return hadamard