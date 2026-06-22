import torch

def initialize_weights(fan_in, fan_out, method):
    """
    Returns: tensor of shape (fan_out, fan_in) with initialized weights
    """
    weights = torch.empty(fan_out, fan_in)
    
    if method == "xavier_uniform":
        low =-1*((6/(fan_in+fan_out))**.5)
        high = ((6/(fan_in+fan_out))**.5)
        weights = weights.uniform_(low,high)

    elif method == "xavier_normal":
        mean = 0
        std = ((2/(fan_in+fan_out))**.5)
        weights = weights.normal_(mean,std)
        
    elif method == "he_uniform":
        low = -1*((6/fan_in)**.5)
        high = ((6/fan_in)**.5)
        weights = weights.uniform_(low,high)
        
    elif method == "he_normal":
        mean = 0
        std = ((2/fan_in)**.5)
        weights = weights.normal_(mean,std)
        
    else:
        raise ValueError

    return weights