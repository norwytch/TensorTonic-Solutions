import numpy as np

def dropout(X, mask, drop_prob, mode):
    """
    Returns: 2D list with values rounded to 4 decimal places.
    """
    X = np.array(X, dtype=float)
    mask = np.array(mask, dtype=float)
    
    if mode=='test':
        return X

    if mode=='train':
        return np.round((X*mask)/(1-drop_prob),4)