import numpy as np

def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """

    v_array = np.array(v)

    l1 = np.linalg.norm(v_array, ord=1)
    l2 = np.linalg.norm(v_array, ord=2)
    l_inf = np.linalg.norm(v_array, ord=np.inf)
    
    norm_array = [l1, l2, l_inf]
    return norm_array