import os
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
import pickle

class NTU60Dataset(Dataset):
    def __init__(self, data_path, split="train", transform=None):
        

        self.transform = transform
        if split == "train":
            data_file = os.path.join(data_path, "train_data_joint.npy")
            label_file = os.path.join(data_path, "train_label.pkl")
        elif split == "val":
            data_file = os.path.join(data_path, "val_data_joint.npy")
            label_file = os.path.join(data_path, "val_label.pkl")
        else:
            raise ValueError(f"Unknown split: {split}")

        self.data = np.load(data_file, mmap_mode='r')  # (N, C, T, V, M)
        with open(label_file, "rb") as f:
            _, self.labels = pickle.load(f)
            
        self.classes = list(set(self.labels))
        self.targets = self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index].copy()).float()  # shape: (C, T, V, M)
        y = self.labels[index]
        # print(f"[DEBUG] Before transform: x.shape={x.shape}")
        if self.transform:
            x = self.transform(x)
            # print(f"[DEBUG] After transform: x type={type(x)}, x.shape={x.shape}")
            # print(f'[DEBUG] {self.transform}')
        return index, x, y

class KineticsSkeletonDataset(Dataset):
    """Kinetics-400 pre-processed skeleton dataset."""

    def __init__(self, data_path, split = "train", transform = None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        
        self.data_file = os.path.join(data_path, f'{split}_data_joint.npy')
        self.label_file = os.path.join(data_path, f'{split}_label.pkl')

        self.data = np.load(self.data_file, mmap_mode='r')
        
        with open(self.label_file, 'rb') as f:
            self.label_info = pickle.load(f)
        self.labels = np.array(self.label_info[1])

        self.classes = list(set(self.labels))
        self.targets = self.labels

        assert len(self.data) == len(self.labels), "Data and labels must have the same length"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_numpy = self.data[index][:,:,:,:].copy()
        y = self.labels[index]
        x = torch.from_numpy(data_numpy).float()
        
        if self.transform:
            x = self.transform(x)
            
        return index, x, y