import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    #forward, loss, backwards, step
    num_batches = len(dataloader)
    acc_loss = 0.0
    
    for batch in dataloader:
        optimizer.zero_grad()
        inputs, targets = batch
        output = model(inputs)
        loss = criterion(output, targets)
        acc_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = acc_loss/num_batches
    return avg_loss
