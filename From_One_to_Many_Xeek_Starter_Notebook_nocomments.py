import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset  

import torch.utils.data as data_utils


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import yaml

from src import CVAE_antares as antares 

from src import CVAE_functions as CVAE_fn

torch.manual_seed(1)
torch.cuda.manual_seed(1)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
print("cude, device: ", cuda, device)

data_root = './data/'
filename = "x0_x1_y" 

filename_dataset = f'{data_root}{filename}.pt'
filename_test_dataset = f'{data_root}{filename}_test.pt'

dataset = CVAE_fn.SignalDataset_v4(torch.load(filename_dataset))
dataset_test = CVAE_fn.SignalDataset_v4(torch.load(filename_test_dataset))

print(f"Train dataset is loaded successfully and has length: {len(dataset)}")
print(f"Test dataset is loaded successfully and has length: {len(dataset_test)}")

dataloader = DataLoader(dataset, batch_size = 100, shuffle=True)

test_loader = DataLoader(dataset_test, batch_size=10, shuffle=True)

dataset_size = len(dataset)

test_size = len(test_loader)

print(f"Dataset size: {dataset_size}")
print(f"Test size: {test_size}")


x,y = next(iter(dataloader))        

num_samples = 30

CVAE_fn.plot_samples(x, y, num_samples = num_samples)





model_parameters = {
    "number_of_points": 50,
    "bias": True,
    "in_channels1": 1,
    "out_channels1": 32,
    "kernel_size1": [16, 2],
    "out_channels2": 64,
    "kernel_size2": [8, 1]
}

with open('src/config_model.yaml', 'w') as f:
    yaml.dump(model_parameters, f)
    
    
number_of_points = 50
input_shape = (1, number_of_points, 2)
cond_shape = (number_of_points,)



hyperparameters = {
    "latent_dim": 6,
    "lr" : 0.001,
    "batch_size" : 100,
    "beta" : 1,
    "wx": 0.02,
    "wy": 0.01,
    "num_epochs":20
}

latent_dim, lr, batch_size, beta, wx, wy, num_epochs = hyperparameters.values()
wx = wx * beta
wy = wy * beta

cvae = antares.CVAE(latent_dim).to(device)

model = cvae
optimizer = optim.Adam(model.parameters(), lr = lr)

print("Current CVAE configuration: ")
model








for epoch in range(1, num_epochs + 1):
    CVAE_fn.train_cvae(model, dataloader, optimizer, beta, wx, wy, epoch, device)
    
CVAE_fn.test_cvae(model, test_loader,beta, wx, wy, device)


'''
YOUR CODE HERE

'''


num_samples = 30 

x_batch, y_batch = next(iter(dataloader))

y_idx = np.random.randint(batch_size)

given_y = y_batch[y_idx].unsqueeze(0).to(device)

x_output, y_output = CVAE_fn.generate_samples(model, num_samples, given_y, input_shape, device)

CVAE_fn.plot_samples_stacked(x_output.cpu(), y_output.cpu())


    
'''
x_outputs = torch.cat([...], dim=0)   
y_outputs = torch.cat([...], dim=0) 
'''



ds = data_utils.TensorDataset(x_output,y_output)
torch.save(ds, 'result/result.pt') 


'''
hyperparameters = {
    "latent_dim":  ,
    "lr" :   ,
    "batch_size" :   ,
    "beta" :  ,
    "wx":  ,
    "wy":  ,
    "num_epochs":
}
'''

with open('result/best_hyperparameters.yaml', 'w') as f:
    yaml.dump(hyperparameters, f)
 






