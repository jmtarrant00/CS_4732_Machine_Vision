import torch
import torch.nn as nn

from math import floor

import torch
import torch.nn as nn

class VAE_NETWORK(nn.Module):
    def __init__(self, depth, width, height, latent_size):
        super().__init__()

        self.depth = depth
        self.width = width
        self.height = height
        self.latent_size = latent_size

        self.encoder_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=3, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(3),
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self._get_output_size()

        self.mean_layer = nn.Linear(self.output_size, latent_size)
        self.log_var_layer = nn.Linear(self.output_size, latent_size)

        self.decoder_conv_layers = nn.Sequential(

            nn.ConvTranspose2d(in_channels=latent_size, out_channels=6, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=4, stride=2),
            nn.Sigmoid()
        )

    def _get_output_size(self):
        x = torch.randn(1, self.depth, self.width, self.height)
        x = self.encoder_conv_layers(x)
        x = x.view(x.size(0), -1)
        self.output_size = x.size(1)

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mean + epsilon * std
        return z

    def forward(self, X):
        encoded = self.encoder_conv_layers(X)
        encoded = encoded.view(encoded.size(0), -1)
        mean = self.mean_layer(encoded)
        log_var = self.log_var_layer(encoded)
        Z = self.reparameterization(mean, log_var)
        reconstructed_data = self.decoder_conv_layers(Z)
        return reconstructed_data, mean, log_var

    
    
    def reconstuction_loss(self, input, reconstructed):
        return nn.functional.mse_loss(input, reconstructed)
    
    def KL_Divergence(self, mean, log_var):
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    def loss_fn(self, input, reconstructed, mean, log_var):
        reconstruction_loss = self.reconstuction_loss(input, reconstructed)
        KL_Divergence = self.KL_Divergence(mean, log_var)

        return reconstruction_loss + KL_Divergence
