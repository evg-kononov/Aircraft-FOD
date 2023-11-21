import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root_dir, num_classes=None, transform=None, target_transform=None):
        self.labels = []
        self.img_paths = []
        for path, folder, files in os.walk(root_dir):
            if len(folder) == 0:
                folder = path.split("/")[-1]
                label = float(folder[0])
            for file in files:
                self.labels.append(label)
                self.img_paths.append(os.path.join(path, file))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        label = torch.tensor([self.labels[idx]])
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