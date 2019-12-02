import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs
    