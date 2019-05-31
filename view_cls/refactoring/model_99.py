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
from .train_snippets import train_model
import ipdb
from .visualize_images import imshow
import time
import logging
from .dataset_dataloaders import dataloaders


N_CLASSES = 3

model_ft = models.resnet50(pretrained=True)

ct = 0
for child in model_ft.children():
    ct += 1
    if ct <= 7:
        print('freezing',child)
        for param in child.parameters():
            param.requires_grad = False
    else:
        print(child,'remains unfrozen')

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, N_CLASSES)

