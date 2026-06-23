import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    T = scores.shape[-1]
    mask = np.triu(np.full((T,T), mask_value), k=1)
    scores = np.where(mask, mask_value, scores)
    return scores