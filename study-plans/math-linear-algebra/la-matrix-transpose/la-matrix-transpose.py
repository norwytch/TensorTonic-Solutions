import numpy as np

def matrix_transpose(A):
    """
    Returns: ndarray, the transpose of A.
    """
    A_matrix = np.array(A)
    AT = np.zeros((A_matrix.shape[1], A_matrix.shape[0]))
    for i in range(A_matrix.shape[0]):
        for j in range(A_matrix.shape[1]):
            AT[j,i] = A_matrix[i,j]

    return AT