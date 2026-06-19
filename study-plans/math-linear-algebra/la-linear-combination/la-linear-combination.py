import numpy as np

def linear_combination(vectors, coefficients):
    """
    Returns: float64 array, the weighted sum of vectors.
    """
    vectors_array = np.array(vectors)
    coefficients_array = np.array(coefficients)
    k = len(vectors_array)
    w = 0
    for i in range(k):
        w += coefficients_array[i]*vectors_array[i]

    return w
    