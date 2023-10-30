import argparse
import time
import torch
import torchvision
import torchvision.transforms as transforms
import vae_network

from sys import exit
from time import time
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

### ARGUMENT PARSING ###
 
# Set up the argument parser
parser = argparse.ArgumentParser()

# Set the arguments for the parser
parser.add_argument('-e', '--epochs', type=int, help='Specify the number of epochs for the training loop to run.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='Specify the learning rate for the training phase. Default value is 0.001.')
parser.add_argument('-c', '--checkpoint', type=str, help='Give the path for a checkpoint to load a model from.')

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

### MODEL SET-UP ###

# Initialize the model
if not args.checkpoint:
    model = vae_network.VAE_NETWORK(depth=depth, width=width, height=height, latent_size=16)
else:
    kwargs, model = torch.load('model.pth')
    model = vae_network.VAE_NETWORK(**kwargs)
    model.load_state_dict(model)

print(model)

### TRAINING PHASE ###



### TESTING PHASE