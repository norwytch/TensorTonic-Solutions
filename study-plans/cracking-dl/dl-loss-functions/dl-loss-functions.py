import numpy as np

def loss_functions(y_true, y_pred, loss_type):
    """
    Returns: Loss value as a float, rounded to 4 decimal places.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_pred)
    loss = None
    
    if loss_type=='mse':
        loss = np.mean((y_true - y_pred)**2)
        loss = np.round(loss.item(),4)
        
    if loss_type=='bce':
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        term = y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred)
        loss = -1*np.mean(term)
        loss = np.round(loss.item(),4)
        
    if loss_type=='cce':
        correct  = y_pred[np.arange(n), y_true]
        m = y_pred.max(axis=1, keepdims=True)
        logsumexp = m.squeeze() + np.log(np.sum(np.exp(y_pred - m), axis=1 ))
        per_sample = -(correct - logsumexp)
        loss = per_sample.mean()
        loss = np.round(loss.item(),4)
    
    if loss_type=='hinge':
        per_sample = np.maximum(0, 1-y_true*y_pred)
        loss = np.mean(per_sample)
        loss = np.round(loss.item(),4)
        
    return loss