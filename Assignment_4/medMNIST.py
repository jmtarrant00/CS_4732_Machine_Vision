import argparse
import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from medmnist import INFO, evaluator

print(f'MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}')

data_flag = 'pneumoniamnist'
download = True

NUM_EPOCHS = 3

BATCH_SIZE = 128

lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = info['n_classes']


DataClass = getattr(medmnist, info['python_class'])

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])




# Load the Data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

pil_dataset = DataClass(split='train', download=download)


# Set up the dataloaders
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(pil_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
