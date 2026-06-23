import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    """
    Returns: (features, label) from __getitem__ where features is float32 (D,) and label is float32 (1,)
    """

    def __init__(self, data, label_col):
        self.data = data
        self.label_col = label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        label = [row[self.label_col]]
        features = row[0:self.label_col] + row[self.label_col+1:]

        features = torch.tensor(features, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        
        return features,label
