import numpy as np

def matrix_vector_multiply(A, x):
    """
    Returns: 1-D float64 array, the product A @ x.
    """
    A, x = np.array(A), np.array(x)
    m, n = A.shape[0], A.shape[1]
    product = np.zeros((m))
    
    for i in range(m):
        for j in range(n):
            product[i] += A[i,j]*x[j]

    return product