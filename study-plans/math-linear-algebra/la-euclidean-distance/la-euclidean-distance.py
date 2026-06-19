import numpy as np

def euclidean_distance(x, y):
    """
    Returns: float, the Euclidean distance between x and y.
    """
    x_array, y_array = np.array(x), np.array(y)
    return np.linalg.norm(x_array-y_array)