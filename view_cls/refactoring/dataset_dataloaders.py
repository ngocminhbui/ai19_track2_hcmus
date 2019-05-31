import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
#import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from train_snippets import train_model
import ipdb
from .visualize_images import imshow
import time
import logging
from .transforms import data_transforms, mask_tensor


## PREPARING DATASET

dataset = ImageFolder('/home/hthieu/AICityChallenge2019/data/Track2Data/vehicle_views_v3/',
                      transform=data_transforms['train'], target_transform=lambda x: mask_tensor[x])
print(dataset)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_dataset.dataset = copy.copy(dataset) #copy, so it can have other transformation.
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

dataloaders = {}
dataloaders['train'] = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
dataloaders['val'] = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)

logging.info(dataloaders['train'].dataset)
logging.info('*'*20)
logging.info(dataloaders['val'].dataset)
logging.info('logged')

from PIL import Image

class CSVDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, only_filename=True):
        self.root_dir = root_dir
        self.transform = transform
        self.csv = np.genfromtxt(csv_file, delimiter=',', dtype='str')
        self.only_filename = only_filename

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if self.only_filename == False:
            img_name = os.path.join(self.root_dir,
                                    self.csv[idx][1])
        else:
            img_name = os.path.join(self.root_dir,
                                    self.csv[idx])

        image = Image.open(img_name)


        if self.transform:
            sample = self.transform(image)

        return sample
