import torch
import torch.nn as nn

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience):
    """
    Returns: dict with 'train_losses' (list), 'val_losses' (list), 'stopped_epoch' (int, 1-indexed)
    """
    train_losses = []
    val_losses = []
    batch_num = len(train_loader)
    best = float('inf')
    counter = 0

    for epoch in range(max_epochs):
        train_sum = 0.0
        val_sum = 0.0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_sum+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for inputs, targets in val_loader:
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_sum+=loss.item()
        avg_train_loss = train_sum/batch_num
        avg_val_loss = val_sum/len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        if avg_val_loss < best:
            best = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                return_dict = {'train_losses': train_losses, 'val_losses': val_losses, 'stopped_epoch': epoch+1}
                return return_dict
    return_dict = {'train_losses': train_losses, 'val_losses': val_losses, 'stopped_epoch': max_epochs}
    return return_dict