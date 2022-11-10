import torch
from torch.utils.data import Dataset


class PROJDataset(Dataset):
    def __init__(self, proj_data):
        self.X = proj_data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        data = self.X.iloc[ind, :-1]
        target_ = self.X.iloc[ind, -1:]

        if target_ is not None:
            return torch.tensor(data.values), torch.tensor(target_.values)
        else:
            return torch.tensor(data)
