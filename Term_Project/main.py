import argparse
import matplotlib.pyplot as plt
import numpy as np
import piq
import time
import torch
import torch.amp
import torchvision.transforms as transforms
import vae_network as vae_network

from sys import exit
from time import time
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
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
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
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
num_train_images = len(dataset)
train_size = int(num_train_images * 0.6)
val_size = int(num_train_images * 0.1)
test_size = num_train_images - train_size - val_size

train_set, validation_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size = 250, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=35, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

### MODEL SET-UP ###

# Initialize the model
if not args.checkpoint:
    model = vae_network.VAE_NETWORK(depth=depth, width=width, height=height)
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
if not args.checkpoint:
    # Set the autocast type based on the device
    if device == 'cuda':
        autocast_type = torch.float16
    else:
        autocast_type = torch.bfloat16

    # Set up a list for losses
    loss_list = []

    pbar_update = 100 / (args.epochs * 10)
    print(pbar_update)


    first_validation = True
    completed_epochs = 0
    start_time = time()
    with tqdm(total=100) as pbar:
        for epoch in range(args.epochs):
            model.train()
            for train_images, _ in train_loader:
                with torch.autocast(device_type=device, dtype=autocast_type):
                    # Move the tensors to device
                    train_images = train_images.to(device)
                
                    # Run the train_images through the network
                    outputs, mean, log_var = model(train_images)
                    outputs = torch.clamp(outputs, min=0, max=1)

                    
                    loss = model.loss_fn(train_images, outputs, mean, log_var)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                pbar.update(pbar_update)

            loss_list.append(loss.item())


            # Validation
            
            model.eval()
            with torch.no_grad():
                for valid_images, _ in validation_loader:
                    with torch.autocast(device_type=device, dtype=autocast_type):
                        # Move the tensors to device
                        valid_images = valid_images.to(device)
                    
                        # Run the train_images through the network
                        valid_outputs, mean, log_var = model(valid_images)
                        valid_outputs = torch.clamp(valid_outputs, min=0, max=1)
                        validation_loss = model.loss_fn(valid_images, valid_outputs, mean, log_var)

                if first_validation:
                    old_validation_loss = validation_loss
                    first_validation = False

                if validation_loss > old_validation_loss:
                    break
                else:
                    old_validation_loss = validation_loss
            completed_epochs += 1

            
    

    end_time = time()
    # Print out some training stats
    print('\n----- TRAINING STATS -----')
    print(f'Training time: {(end_time - start_time):.2f} secs')
    print(f'Completed {completed_epochs} epochs')
    print(f'Max Loss: {np.max(loss_list)}')
    print(f'Min Loss: {np.min(loss_list)}')
    print(f'Average Loss: {np.average(loss_list):.3f}\n')


    # Graph the loss if asked for 
    if args.graph:
        plt.figure()
        plt.plot([x for x in range(len(loss_list))], loss_list)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.savefig('loss_plot.svg')


### TESTING PHASE ###

# Initalize Testing Loss
testing_loss = 0

# Initalize some lists to hold various testing stats
testing_loss_list = []
psnr_list = torch.tensor([],dtype=torch.float, device=device)
brisque_list = torch.tensor([],dtype=torch.float, device=device)
ssim_list = torch.tensor([],dtype=torch.float, device=device)

# Set the model into eval mode
model.eval()

# Turn off the gradients
with torch.no_grad():
    for test_images, _ in test_loader:
        with torch.autocast(device_type=device):
            # Move the image tensors to the device
            test_images = test_images.to(device)

            # Run the images through the newtork
            test_outputs, mean, log_var = model(test_images)
            testing_loss = model.loss_fn(test_images, test_outputs, mean, log_var)

            test_outputs = torch.clamp(test_outputs, min=0)

            # Calculate some testing stats
            for i in range(test_outputs.shape[0]):
                psnr = piq.psnr(test_outputs[i:i+1], test_images[i:i+1], convert_to_greyscale=True)
                psnr_list = torch.cat([psnr_list, psnr.unsqueeze(0)])

                brisque = piq.brisque(test_outputs, kernel_size=7)
                brisque_list = torch.cat([brisque_list, brisque.unsqueeze(0)])

                ssim = piq.ssim(test_images, test_outputs, kernel_size=3)
                ssim_list = torch.cat([ssim_list, ssim.unsqueeze(0)])
        
        testing_loss_list.append(testing_loss.item())
        save_image(test_images[0], 'output_images/original.svg')
        save_image(test_outputs[0], 'output_images/denoised.svg')



# Print out some testing stats
print('\n----- TESTING STATS -----')
print(f'Peak Signal-to-Noise Ratio (PSNR):  \t{torch.mean(psnr_list)}')
print(f'BRISQUE Score:                      \t{torch.mean(brisque_list)}')
print(f'Structural Similarity Index Measure:\t{torch.mean(ssim_list)}')


# Save the model 
torch.save([model.kwargs, model.state_dict()], 'model.pth')