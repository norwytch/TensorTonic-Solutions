import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    num_batches = len(dataloader)
    accum_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        inputs, targets = batch
        output = model(inputs)
        loss = criterion(output, targets)
        accum_loss += loss
        loss.backward()
        optimizer.step()

    avg_loss = accum_loss/num_batches
    return avg_loss
        
