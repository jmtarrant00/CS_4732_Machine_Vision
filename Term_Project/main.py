import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.amp
import torchvision
import torchvision.transforms as transforms
import vae_network as vae_network

from sys import exit
from time import time
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler
from tqdm import tqdm

# Print out the device the model is running on 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'PyTorch is running on {device}')

### ARGUMENT PARSING ###
 
# Set up the argument parser
parser = argparse.ArgumentParser()

# Set the arguments for the parser
parser.add_argument('-e', '--epochs', type=int, help='Specify the number of epochs for the training loop to run.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                    help='Specify the learning rate for the training phase. Default value is 0.001.')
parser.add_argument('-c', '--checkpoint', type=str, help='Give the path for a checkpoint to load a model from.')
parser.add_argument('-g', dest='graph', action='store_true', help='Saves a graph of the loss over epochs')

# Parse the arguments and get the namespace for them
args = parser.parse_args()


### DATA PREPROCESSING ###

# Set up a transform for importing the data
transform = transforms.Compose([
    transforms.ToTensor()
])

# Import the dataset
dataset = ImageFolder('dataset/', transform=transform)

# Get depth, width and height
image, label = dataset[0]
depth, width, height = image.shape

# Split the data into training and testing sets
num_images = len(dataset)
train_size = int(num_images * 0.6)
val_size = int(num_images * 0.1)
test_size = num_images - train_size - val_size

train_set, validation_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size = 250, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=35, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

### MODEL SET-UP ###

# Initialize the model
if not args.checkpoint:
    model = vae_network.VAE_NETWORK(depth=depth, width=width, height=height, latent_size=16)
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()
else:
    kwargs, model = torch.load('model.pth')
    model = vae_network.VAE_NETWORK(**kwargs)
    model.load_state_dict(model)

# Move the model to the gpu if available
model.to(device)

# Save the model architecture to a file
with open('model_architecture.txt', 'w') as architecture_file:
    print(model, file=architecture_file)
architecture_file.close()


### TRAINING PHASE ###

# Set the autocast type based on the device
if device == 'cuda':
    autocast_type = torch.float16
else:
    autocast_type = torch.bfloat16

# Set up a list for losses
loss_list = []

pbar_update = args.epochs / 10

start_time = time()
with tqdm(total=100) as pbar:
    for epoch in range(args.epochs):
        for images, labels in train_loader:
            with torch.autocast(device_type=device, dtype=autocast_type):
                # Move the tensors to device
                images = images.to(device)
                labels = labels.to(device)
            
                # Run the images through the network
                outputs, mean, log_var = model(images)
                loss = model.loss_fn(images, outputs, mean, log_var)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.update(pbar_update)
        loss_list.append(loss.item())
        


end_time = time()
# Print out some training stats
print('\n----- TRAINING STATS -----')
print(f'Training time: {(end_time - start_time):.2f}')
print(f'Max Loss: {np.max(loss_list)}')
print(f'Min Loss: {np.min(loss_list)}')
print(f'Average Loss: {np.average(loss_list):.3f}\n')


# Graph the loss if asked for 
if args.graph:
    plt.figure()
    plt.plot([x for x in range(args.epochs)], loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.savefig('loss_plot.svg')

### TESTING PHASE