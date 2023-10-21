import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, labels_file, root_dir, num_classes=None, transform=None, target_transform=None):
        self.labels = pd.read_csv(labels_file)
        self.num_classes = num_classes
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, str(self.labels.iloc[idx, 0]) + ".npy")
        image = np.load(img_path)
        label = self.labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label, self.num_classes)
        return image, label


def transform():
    pass


def target_transform(label, num_classes):
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    return F.one_hot(label, num_classes)


def generator(dataloader):
    while True:
        for batch in dataloader:
            yield batch