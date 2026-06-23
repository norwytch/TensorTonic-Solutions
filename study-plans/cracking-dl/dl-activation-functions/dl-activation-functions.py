import numpy as np

def activation_functions(x, activation):
    """
    Returns: list
    """
    output = None
    dx = None
    if activation=='relu':
        output = np.maximum(0,x)
        if x>0:
            dx = 1
        else:
            dx = 0
    if activation=='sigmoid':
        output = 1/(1+np.exp(-x))
        dx = 1/(1+np.exp(-x)) * (1 - 1/(1+np.exp(-x)))
    if activation=='tanh':
        numer = (np.exp(x)-np.exp(-x))
        denom = (np.exp(x)+np.exp(-x))
        output = numer/denom
        dx = 1 - ((numer/denom)**2)
    if activation=='leaky_relu':
        if x>0:
            output = x
            dx = 1
        else:
            output = .01*x
            dx = .01
    if activation=='gelu':  
        output = .5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*(x**3))))
        g_prime = np.sqrt(2/np.pi)*(1+3*.044715*x**2)
        dx = .5*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*(x**3)))) + .5*x*(1-np.tanh(np.sqrt(2/np.pi)*(x+.044715*(x**3)))**2)*g_prime
    if activation=='swish':
        output = x*(1/(1+np.exp(-x)))
        dx = (1/(1+np.exp(-x))) + x*(1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))
    output = float(output)
    dx = float(dx)
    return [np.round(output, 4), np.round(dx, 4)]