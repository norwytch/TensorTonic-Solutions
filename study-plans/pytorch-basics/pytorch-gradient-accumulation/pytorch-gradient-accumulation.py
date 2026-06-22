import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    """
    Returns: tuple of (updated_weights_list, last_avg_gradient_list)
    """
    w = torch.tensor(w_init, requires_grad=True, dtype=torch.float32)
    acc_grad = 0.0
    acc_loss = 0.0
    batch_counter = 0
    avg_grad = 0
    for inputs, targets in micro_batches:
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        predictions = (w@inputs)
        loss = (predictions - targets)**2
        loss.backward()
        batch_counter +=1
        if batch_counter == accum_steps:
            avg_grad = w.grad/accum_steps
            with torch.no_grad():
                w -= lr*avg_grad
            w.grad = None
            batch_counter = 0
    return (w.tolist(), avg_grad.tolist())
