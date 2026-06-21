import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    counter = 0
    total_loss = 0.0
    for inputs, targets in dataloader:
        counter+=1
        optimizer.zero_grad()
        outputs = model(inputs)
        scores = criterion(outputs, targets)
        total_loss += scores
        scores.backward()
        optimizer.step()
    return total_loss/counter
