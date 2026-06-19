import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a_array, b_array = np.array(a), np.array(b)
    if np.linalg.norm(a_array) == 0 or np.linalg.norm(a_array) == 0:
        return 0
    else:    
        cos_sim = np.dot(a_array, b_array) / (np.linalg.norm(a_array) * np.linalg.norm(b_array))
    return cos_sim
