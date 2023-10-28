import torch
import torch.nn as nn

from math import floor

class VAE_NETWORK(nn.Module):
    def __init__(self, depth, width, height, latent_size) -> None:
        super().__init__(VAE_NETWORK, self)

        self.kwargs = {
            'depth': depth,
            'width': width,
            'height': height,
            'latent_size': latent_size
        }

        self.depth = depth
        self.width = width
        self.height = height
        self.latent_size = latent_size
        self.conv_output_size = 0

        self.encoder_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.depth, out_channels=3, kernel_size=16, stride=8),
            nn.ReLU(),
            nn.MaxPool2d(),
            nn.BatchNorm2d(3),
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=8, stride=4),
            nn.ReLU()
        )

        self.decoder_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=3, out_channels=self.depth, kernel_size=16, stride=8),
            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.encoder_conv_layers(X)
        X = self.decoder_conv_layers(X)
        return(X)
    
    
    def reconstuction_loss(self, input, reconstructed):
        return nn.functional.mse_loss(input, reconstructed)
    
    def KL_Divergence(self, mean, log_var):
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    def loss_fn(self, input, reconstructed, mean, log_var):
        reconstruction_loss = self.reconstuction_loss(input, reconstructed)
        KL_Divergence = self.KL_Divergence(mean, log_var)

        return reconstruction_loss + KL_Divergence
