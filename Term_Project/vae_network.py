import torch
import torch.nn as nn

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class VAE_NETWORK(nn.Module):
    def __init__(self, depth, width, height, latent_size):
        super().__init__()

        self.depth = depth
        self.width = width
        self.height = height
        self.latent_size = latent_size

        self.encoder_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=3, kernel_size=4, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=1, bias=False),
            nn.ReLU()
        )

        self._get_output_size()

        self.mean_layer = nn.Linear(self.output_size, self.output_size)
        self.log_var_layer = nn.Linear(self.output_size, self.output_size)

        self.decoder_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=2, stride=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=3, out_channels=depth, kernel_size=4, stride=2, bias=False),
            nn.Sigmoid()
        )


    def _get_output_size(self):
        x = torch.randn(1, self.depth, self.width, self.height)
        x = self.encoder_conv_layers(x)
        x = x.view(x.size(0), -1)
        self.output_size = x.size(1)

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var).to(device)
        epsilon = torch.randn_like(std).to(device)
        z = mean + epsilon * std
        return z

    def forward(self, X):
        encoded = self.encoder_conv_layers(X)
        _ , depth, width, height = encoded.shape
        encoded = nn.Flatten()(encoded)  

        mean = self.mean_layer(encoded).to(device)
        log_var = self.log_var_layer(encoded)

        Z = self.reparameterization(mean, log_var)

        Z = nn.Unflatten(1, (depth, width, height))(Z)
        reconstructed_data = self.decoder_conv_layers(Z)
        return reconstructed_data, mean, log_var

    
    
    def reconstuction_loss(self, input, reconstructed):
        return nn.functional.mse_loss(input, reconstructed).to(device)
    
    def KL_Divergence(self, mean, log_var):
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    def loss_fn(self, input, reconstructed, mean, log_var):
        reconstruction_loss = self.reconstuction_loss(input, reconstructed)
        KL_Divergence = self.KL_Divergence(mean, log_var)

        return reconstruction_loss + KL_Divergence
