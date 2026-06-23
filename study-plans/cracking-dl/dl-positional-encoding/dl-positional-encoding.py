import numpy as np

def rope(x, mode="forward", d_output=None):
    """
    Returns: Dict with "rotated", "cos_pe", "sin_pe" (and "dx" in backward mode).
    All values rounded to 4 decimal places.
    """
    x = np.array(x)
    T, d = x.shape

    theta = np.zeros((T, d//2))
    rotated = np.zeros((T,d))
    positions = np.arange(T)[:, None]
    pairs = np.arange(d//2)[None,:]
    theta = positions/(10000**(2*pairs/d))
    cos_pe = np.cos(theta)
    sin_pe = np.sin(theta)
    x_even = x[:,0::2]
    x_odd = x[:,1::2]
    rot_even = x_even * cos_pe - x_odd * sin_pe
    rot_odd = x_even * sin_pe + x_odd * cos_pe
    rotated[:,0::2] = rot_even
    rotated[:,1::2] = rot_odd


    if mode=='backward':
        d_output = np.array(d_output)
        dx = np.zeros((T,d))
        d_even = d_output[:,0::2]
        d_odd = d_output[:,1::2]
        dx_even = d_even*cos_pe + d_odd*sin_pe
        dx_odd = -d_even*sin_pe + d_odd*cos_pe
        dx[:, 0::2] = dx_even
        dx[:, 1::2] = dx_odd
        return {"rotated": np.round(rotated, 4), "cos_pe": np.round(cos_pe, 4), "sin_pe": np.round(sin_pe, 4), "dx": np.round(dx, 4)}

    return {"rotated": np.round(rotated, 4), "cos_pe": np.round(cos_pe, 4), "sin_pe": np.round(sin_pe, 4)}
    
        