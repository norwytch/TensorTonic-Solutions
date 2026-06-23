import numpy as np

def relu(z):
    return np.maximum(0,z)

def forward_pass(x, weights, biases):
    """
    Returns: Dict with "activations" and "pre_activations", values rounded to 4 decimals.
    """
    x = np.array(x)
    a = x
    pre_activations = []
    activations = []
    activations.append(a)
    max_len = len(weights)
    counter = 0
    for W, b in zip(weights, biases):
        z = W@a + b
        z = np.round(z, 4)
        pre_activations.append(z)
        if counter != max_len-1:
            a = relu(z)
            a = np.round(a, 4)
            activations.append(a)
            counter +=1
        else:
            a = z
            activations.append(a)
    
    my_dict = {'activations': activations, 'pre_activations': pre_activations}
    return my_dict