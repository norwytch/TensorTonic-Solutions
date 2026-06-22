import torch
import torch.nn as nn

def train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, num_epochs):
    """
    Returns: dict with 'losses' (list of per-epoch avg loss) and 'lrs' (list of learning rate per epoch)
    """
    batch_counter = 0
    current_loss = 0.0
    loss_list = []
    lr_list = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        current_lr = scheduler.get_last_lr()[0]
        lr_list.append(current_lr)
        for inputs, targets in dataloader:
            batch_counter +=1
            optimizer.zero_grad()
            outputs = model(inputs)
            scores = criterion(outputs, targets)
            batch_loss = scores.item()
            epoch_loss += batch_loss
            current_loss += scores.item()
            scores.backward()
            optimizer.step()
        avg_loss=epoch_loss/batch_counter
        loss_list.append(avg_loss)
        scheduler.step()
        batch_counter=0
    epoch_dict = {"losses": loss_list, "lrs": lr_list}
    return epoch_dict
