import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
#import matplotlib.pyplot as plt
import time
import os import copy
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from train_snippets import train_model
import ipdb
from visualize_images import imshow
import time
import logging
logging.basicConfig(filename='./log/train_{}.log'.format(time.time()), level=logging.INFO,
                    filemode='w', format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
#DATASET & MODEL
from dataset_dataloaders import dataloaders
from model import model_ft

#TRAINING SETTINGS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

criterion = nn.BCEWithLogitsLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters())

# Decay LR by a factor of 0.1 every 10 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft,mode='min',factor=0.1,
                                                    patience=10,threshold=0.0001)

# START TRAINING
model_ft = train_model(model_ft, criterion, optimizer_ft, lr_scheduler,dataloaders=dataloaders,
                                              num_epochs=100, device=device)
