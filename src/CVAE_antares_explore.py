import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader 

import numpy as np
import matplotlib.pyplot as plt


# define the encoder network
class Encoder(nn.Module):
    def __init__(self, latent_dim, in_channels1, out_channels1, kernel_size1, out_channels2, kernel_size2, bias, dimInt):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels1, out_channels1, kernel_size=kernel_size1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=kernel_size2, bias=bias)
        self.fc1_mean = nn.Linear(out_channels2 * dimInt, latent_dim)
        self.fc1_logvar = nn.Linear(out_channels2 * dimInt, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the feature map
        
        z_mean = self.fc1_mean(x)
        z_logvar = self.fc1_logvar(x)
        
        return z_mean, z_logvar

# define the sampling layer
class Sampling(nn.Module):  #The same approach as reparametrization trick
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, z_mean, z_logvar):
        batch_size, latent_dim = z_mean.size()
        epsilon = torch.randn(batch_size, latent_dim).to(z_mean.device)
        std = torch.exp(0.5 * z_logvar)
        z = z_mean + std * epsilon
        return z

# define the decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim, number_of_points, out_channels2, dimInt, out_channels1, in_channels1, kernel_size1, kernel_size2, bias):
        super(Decoder, self).__init__()
        self.out_channels2 = out_channels2
        self.dimInt = dimInt
        self.fc1 = nn.Linear(latent_dim + number_of_points, out_channels2 * dimInt)
        self.conv_transpose1 = nn.ConvTranspose2d(out_channels2, out_channels1, kernel_size=kernel_size2, bias=bias)
        self.conv_transpose2 = nn.ConvTranspose2d(out_channels1, in_channels1, kernel_size=kernel_size1, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        x = F.relu(self.fc1(x))
        batch_size, latent_dim = z.size()
        x = torch.reshape(x, (batch_size, self.out_channels2, self.dimInt, 1))  # Reshape to (batch_size, channels, height, width)
        x = self.conv_transpose1(x)
        x = self.conv_transpose2(x)
        x = self.relu(x)
        return x

# define the CVAE model
class CVAE(nn.Module):
    def __init__(self,config,latent_dim = 8):
        super(CVAE, self).__init__()
        # Extract the parameters from the config
        number_of_points = config["number_of_points"]

        bias = config["bias"]
        in_channels1 = config["in_channels1"]
        out_channels1 = config["out_channels1"]
        kernel_size1 = config["kernel_size1"]
        out_channels2 = config["out_channels2"]
        kernel_size2 = config["kernel_size2"]
        #latent_dim = config["latent_dim"]

        # define the input dimensions
        input_shape = (1, number_of_points, 2)
        cond_shape = (number_of_points,)
        dimInt = (number_of_points-kernel_size1[0]+1-kernel_size2[0]+1)
        
        self.encoder = Encoder(latent_dim, in_channels1, out_channels1, kernel_size1, out_channels2, kernel_size2, bias, dimInt)
        self.sampling = Sampling()
        self.decoder = Decoder(latent_dim, number_of_points, out_channels2, dimInt, out_channels1, in_channels1, kernel_size1, kernel_size2, bias)

    def forward(self, x, cond):
        z_mean, z_logvar = self.encoder(x)
        z = self.sampling(z_mean, z_logvar)
        x_recon = self.decoder(z, cond)
        return x_recon, z_mean, z_logvar

