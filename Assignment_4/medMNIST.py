import argparse
import medmnist
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from time import time
from medmnist import INFO, evaluator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')

print(f'MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}')

data_flag = 'pneumoniamnist'
download = True

NUM_EPOCHS = 3

BATCH_SIZE = 128

lr = 0.001

info = INFO[data_flag]
task = info['task']
if task == 'multi-label, binary-class':
    print(task)

n_channels = info['n_channels']
n_classes = len(info['label'])


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


### THE MODEL ###

# Set up the model
class convNet(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3), 
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fully_connected = nn.Sequential(
            nn.Linear (64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.conv_layer_5(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return(x)


# Initialize the model
conv_model = convNet(in_channels=n_channels, num_classes=n_classes)
for param in conv_model.parameters():
    param.requires_grad = True

# Move the model to the gpu if available
if device == 'cuda':
    conv_model.cuda()
    dtype = torch.float16
else:
    dtype = torch.bfloat16

# Define Loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(conv_model.parameters(), lr=lr)


### TRAINING STAGE ###

start_time = time()
loss_list = []

for epoch in range(NUM_EPOCHS):
    for images, targets in train_loader:
        optimizer.zero_grad()
        images = images.to(device)
        targets = targets.to(device)
        
        with torch.autocast(device_type=device, dtype=dtype):
            outputs = conv_model(images)
            loss = loss_fn(outputs, targets.float())
        
        loss.backward()
        
        optimizer.step()
        loss_list.append(loss.item())

print('Finished Training!')
print(f'Average Loss: {np.average(loss_list)}')
print(f'Training Time: {time() - start_time} sec')


