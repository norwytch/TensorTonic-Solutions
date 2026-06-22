import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features, labels, batch_size):
    """
    Returns: a DataLoader that oversamples underrepresented classes
    """
    counts = torch.bincount(labels)
    weight_counts = 1.0/counts[labels]
    dataset_len = len(labels)
    wrs = WeightedRandomSampler(weight_counts, dataset_len)
    dataset_raw = TensorDataset(features, labels)
    bal_dl = DataLoader(dataset_raw, batch_size=batch_size, sampler=wrs)

    return bal_dl