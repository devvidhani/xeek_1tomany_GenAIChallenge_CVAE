#  # From One to Many Challenge - Starter Notebook
# 
#  Challenge link: https://xeek.ai/challenges/from-one-to-many
# 
# 
# ## Background
# 
# The algorithm and challenge idea behind "From One to Many" comes from research done by McAliley and Li (2021) to explore the nonuniqueness of gravity fields.
# 
# Gravitation interpretation is the process of inferring the subsurface structure of the earth by analyzing the gravity field.  This is typically done by measuring the variations in the earth's gravity field using instruments such as gravimeters.  Many factors can contribute to variations in the gravity field, such as variations in rock density, topography, and geologic structure.  Therefore, interpreting the resulting gravity signals can be complex since many subsurface configurations can produce the same signal.
# 
# Using machine learning techniques can help us to explore how the diversity of subsurface configurations gives rise to similar gravity fields.  For example, generative models, such as (conditional) variational autoencoders, (C)VAEs, can be trained to generate synthetic gravity fields that are similar to real-world gravity fields. 
# 
# For this challenge, we will simplify the problem by modeling a signal as a function of two vectors with the given requirements.
# 
# ## The Statement of the Problem
# 
# Let's assume we have given signal Y, which we know is some custom function (for example as the product) of the vectors X0 (decreasing straight line) and X1 (increasing straight line).  Can we generate a complete solution space of combination (X0, X1) producing the same given signal Y?
# 
# We suggest performing the training process with a conditional variational autoencoder (CVAE) using the given architecture described below.  The CVAE is prepared to encode the input signals X0, X1, and Y into a lower-dimensional latent space and to generate diverse combinations (with the greatest possible diversity) of X0 and X1 that can produce similar signals Y. 
# 
# The goal of this challenge is to **customize provided NN architecture and find the optimal values for hyperparameters** to achieve **the mostly complete space of reconstructed X0/X1**.  Remember that X0/X1 should be as close to a straight line as possible, and reconstructed Y should be close to Y given. 
# 
# 
#              
# 
# ## The architecture of the given CVAE   
# 
# 
# The provided architecture is designed as a conditional variational autoencoder (CVAE), including three main components: the Encoder, the Sampling (to perform reparametrization trick), and the Decoder.  Pay attention that the input tensor is concatenated pairs x0 & x1, which passed through the Encoder, and the Decoder is conditioned on the target Y, considered a continuous label.
# 
# Below you can see more detailed specifications of CVAE with enlisted layers of each component.  We invite participants to customize the parameters of this implementation (i.e., input/output sizes) as well as types and amounts of hidden layers, **but preserve interactions of the main components**.  
# 
# To provide more convenient playing with parameters of given layers, you can see all of them in the model_parameters dictionary, which is further stored as config_model.yaml and used during the initialization of the model.  To change the amount and type of layers, you could edit the src/CVAE_architecture.py module.  Please pay attention the functionality to quick start (like train, test, dataset class etc.) provided separately in src/CVAE_function.py module. 
# 
# In particular, the following hyperparameters are supposed to be tuned:
# 
#  - learning_rate: the learning rate for the optimizer used during training
#  - batch_size: the batch size used during training
#  - beta:  regularization parameter for loss function
#  - wx, wy: weigths for reconstruction losses for x, y respectively  
#  - num_epochs: the number of epochs to train for
#  - latent_space_dim: the dimensionality of the latent space used by the CVAE
# 
# 
# ## Evaluation
# 
# The primary objective of the challenge is to design a neural network (specifically a Conditional Variational Autoencoder or CVAE) capable of generating a diverse set of paired straight lines that yield the desired expected result of applying a custom function to each pair.
# In order to evaluate the results, we will consider the following criteria:
# - Straightness of Generated Pairs: It assesses how closely the generated pairs resemble straight lines.
# - Function Result Accuracy: It examines how accurately the custom function applied to the generated pairs aligns with the expected given result.
# - Coverage of Specified Area: The evaluation is mainly based on the extent to which the generated pairs cover the specified range [0, 1] with the spread maximized as possible, but it will only be assessed if both other criteria mentioned above are met.
# 

# ## 0. Environment & settings

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

# Module with CVAE antares architecture
from src import CVAE_antares_explore as antares 

#Module with all related CVAE functions: train, test, etc.
from src import CVAE_functions as CVAE_fn
import itertools
import multiprocessing as mp
from multiprocessing import Pool, set_start_method
import time
import os
import csv
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.spatial import ConvexHull


# set random seeds
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# setup device cuda vs. cpu
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
print("cude, device: ", cuda, device)

# ## 1. Prepare Dataset
# ### 1.1. Load Dataset

data_root = './data/'
filename = "x0_x1_y" 

filename_dataset = f'{data_root}{filename}.pt'
filename_test_dataset = f'{data_root}{filename}_test.pt'

dataset = CVAE_fn.SignalDataset_v4(torch.load(filename_dataset))
dataset_test = CVAE_fn.SignalDataset_v4(torch.load(filename_test_dataset))

print(f"Train dataset is loaded successfully and has length: {len(dataset)}")
print(f"Test dataset is loaded successfully and has length: {len(dataset_test)}")

# get the size of the dataset
dataset_size = len(dataset)

# print the sizes
print(f"Dataset size: {dataset_size}")

# ### 1.2. Plot Random Samples

# x,y = next(iter(dataloader))        

# num_samples = 30

# CVAE_fn.plot_samples(x, y, num_samples = num_samples)

## 2. Model setup and train
### 2.1. Define the model and optimizer

# Please note that the parameters for customizing the model's architecture, such as layer sizes, are created as a **model_parameters** dictionary and stored in a **.yaml** file. Similarly, in the same section, a **hyperparameters** dictionary is created to facilitate the tuning of hyperparameters. The best set of hyperparameters should also be stored in a separate .yaml file (as demonstrated in the code provided in the last part of section #3).   

# Pay attention, that **both files will be required for the final evaluation**, so it is crucial to retain the best configuration during experiments (refer to the 'Submission and Evaluation' section).

# Customize your Model’s Architecture Based on next dictionary 

# Define model architecture parameters and hyperparameters
model_parameters_range = {
    "number_of_points": [50],
    "bias": [True, False],
    "in_channels1": [1],
    "out_channels1": [32, 64],
    "kernel_size1": [[16, 2], [8, 2]],
    "out_channels2": [64, 128],
    "kernel_size2": [[8, 1], [4, 1]]
}

hyperparameters_range = {
    "latent_dim": [4, 6],
    "lr": [0.001, 0.01],
    "batch_size": [100, 200],
    "beta": [0.5, 1, 5],
    # "wx": [0.02, 0.05, 0.1, 0.2, 0.5],
    # "wy": [0.15, 0.2, 0.25, 0.3, 0.5, 0.75],
    # "num_epochs": [10, 20, 30]
    "wx": [0.02, 0.1, 0.5],
    "wy": [0.15, 0.25, 0.75],
    "num_epochs": [10, 30]
}

# model_parameters_range = {
    # "number_of_points": [50],
    # "bias": [True], #, False],
    # "in_channels1": [1],
    # "out_channels1": [32], #, 64],
    # "kernel_size1": [[16, 2]], #, [8, 2]],
    # "out_channels2": [64], #, 128],
    # "kernel_size2": [[8, 1]] #, [4, 1]]
# }
# 
# hyperparameters_range = {
    # "latent_dim": [4], #, 6],
    # "lr": [0.001], #, 0.01],
    # "batch_size": [100], #, 200],
    # "beta": [0.5], #, 1, 5],
    # "wx": [0.02], #, 0.05, 0.1, 0.2, 0.5],
    # "wy": [0.15], #, 0.2, 0.25, 0.3, 0.5, 0.75],
    # "num_epochs": [10] #, 20, 30]
# }

# Initialize best parameter combination and best loss
best_params = None
best_loss = float("inf")

def run_model(params):
    # Grid search
    # for params in itertools.product(*model_parameters_range.values(), *hyperparameters_range.values()):
    #     # Set the model parameters and hyperparameters

    model_parameters = {
        "number_of_points": params[0],
        "bias": params[1],
        "in_channels1": params[2],
        "out_channels1": params[3],
        "kernel_size1": params[4],
        "out_channels2": params[5],
        "kernel_size2": params[6]
    }

    # model_arch_filename = 'src/config_model_{}_{}.yaml'.format(time.time(), os.getpid())

    # with open(model_arch_filename, 'w') as f:
    #     yaml.dump(model_parameters, f)
        
    # Define initial state of hyperparameters
    hyperparameters = {
        "latent_dim": params[7],
        "lr" : params[8],
        "batch_size" : params[9],
        "beta" : params[10],
        "wx": params[11],
        "wy": params[12],
        "num_epochs":params[13]
    }

    # create a PyTorch DataLoader from the dataset
    dataloader = DataLoader(dataset, batch_size = hyperparameters["batch_size"], shuffle=True)

    # create a PyTorch DataLoader from the dataset_test
    test_loader = DataLoader(dataset_test, batch_size=10, shuffle=True)

    # get the size of the dataloader
    test_size = len(test_loader)

    print(f"Test size: {test_size}")

    # # Define the input dimensions
    # number_of_points = 50
    # input_shape = (1, number_of_points, 2)
    # cond_shape = (number_of_points,)



    latent_dim, lr, batch_size, beta, wx, wy, num_epochs = hyperparameters.values()
    wx = wx * beta
    wy = wy * beta

    cvae = antares.CVAE(model_parameters, latent_dim).to(device)

    # Define the model and optimizer
    model = cvae
    optimizer = optim.Adam(model.parameters(), lr = lr)

    #Print model configuration
    print("Current CVAE configuration: ")
    print(model)

    # ### 2.2. Generate samples from the untrained model
    # 
    # Generating samples from an untrained CVAE usually not produce high-quality  outputs since the model has not yet learned the underlying data distribution, but it could serve as a baseline for comparison. This comparison can help you evaluate the progress made during training, assess whether the model has learned meaningful representations, and understand how the quality of the generated samples has improved.

    # x_batch, y_batch = next(iter(dataloader))
    # y_idx = np.random.randint(batch_size)
    # given_y = y_batch[y_idx].unsqueeze(0).to(device) 


    # x_output, y_output = CVAE_fn.generate_samples(model, num_samples, given_y, input_shape, device)

    # CVAE_fn.plot_samples_stacked(x_output.cpu(), y_output.cpu())


    # ### 2.3. Train & test model

    for epoch in range(1, num_epochs + 1):
        CVAE_fn.train_cvae(model, dataloader, optimizer, beta, wx, wy, epoch, device)
        
    total_loss = CVAE_fn.test_cvae(model, test_loader, beta, wx, wy, device)

    # create a PyTorch DataLoader from the dataset
    evalbatchsize = 10
    evaldataloader = DataLoader(dataset, batch_size = evalbatchsize, shuffle=True)

    # Prepare to store the generated samples
    x_outputs = []
    y_outputs = []
    y_true=[]

    # Generate a set of pairs (X0', X1') for multiple given_y's
    num_samples = 30  # You can adjust this number
    input_shape = (1, model_parameters["number_of_points"], 2)
    for i in range(1):  # You can adjust this number
        _, y_batch = next(iter(evaldataloader))
        y_idx = np.random.randint(evalbatchsize)
        given_y = y_batch[y_idx].unsqueeze(0).to(device)
        x_output, y_output = CVAE_fn.generate_samples(model, num_samples, given_y, input_shape, device)
        
        x_outputs.append(x_output)
        y_outputs.append(y_output)
        # y_true.append([given_y]*num_samples)
        y_true.append(given_y.repeat(30, 1))

    # Concatenate all the generated samples
    x_outputs = torch.cat(x_outputs, dim=0)
    y_outputs = torch.cat(y_outputs, dim=0)
    y_true = torch.cat(y_true, dim=0)

    # Prepare to store the Pearson coefficients
    pearson_coefficients = []
    endpoints = []

    # Calculate the Pearson coefficient for each pair of lines
    for i in range(x_outputs.shape[0]):
        x0 = x_outputs[i, 0, :, 0].cpu().detach().numpy()  # Get the X0' line
        x1 = x_outputs[i, 0, :, 1].cpu().detach().numpy()  # Get the X1' line
        pearson_coefficient, _ = pearsonr(np.arange(50), x0)
        pearson_coefficients.append(abs(pearson_coefficient))
        pearson_coefficient, _ = pearsonr(np.arange(50), x0)
        pearson_coefficients.append(abs(pearson_coefficient))

        # Get the endpoints of the lines x0 and x1
        endpoints.append((x0[0], x0[-1]))
        endpoints.append((x1[0], x1[-1]))

    # Convert the list of Pearson coefficients to a numpy array
    pearson_coefficients = np.array(pearson_coefficients)
    avg_pearson_coefficients = np.mean(pearson_coefficients)

    # Print the average and minimum Pearson coefficient
    # print("Average Pearson Coefficient:", np.mean(pearson_coefficients))
    print("Average Pearson Coefficient:", avg_pearson_coefficients)
    print("Minimum Pearson Coefficient:", np.min(pearson_coefficients))

    # Calculate the RMSE
    y_true = y_true.flatten().cpu().detach().numpy()
    y_pred = y_outputs.flatten().cpu().detach().numpy()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("RMSE:", rmse)

    # Calculate the coverage
    x0_min, x0_max = np.min(x0), np.max(x0)
    x1_min, x1_max = np.min(x1), np.max(x1)
    coverage1 = (x0_max - x0_min) * (x1_max - x1_min)
    print("Coverage:", coverage1)

    # Alternate Coverage based on calculate the convex hull
    hull = ConvexHull(endpoints)

    # The area of the convex hull is the area covered by the lines
    area = hull.volume

    print("Area covered by the lines:", area)

    print(str(params), total_loss, avg_pearson_coefficients, rmse, coverage1, area)
    return total_loss, avg_pearson_coefficients, rmse, coverage1, area, params

if __name__ == '__main__':
    # Use the 'spawn' start method
    set_start_method('spawn', force=True)

    # Create a pool of workers

    # Create a pool of workers
    pool = mp.Pool(processes=21)

    try:
        # Use the pool to run the function in parallel
        results = pool.map(run_model, itertools.product(*model_parameters_range.values(), *hyperparameters_range.values()))
    except Exception as e:
        # Log the exception and the current parameter combination
        with open('src/exception_log.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow([str(e), str(random_samples[0])])
        # Raise the exception again
        raise e
    finally:
        # Close the pool
        pool.close()

        # Log the model parameters and results
        log_data = []

        # Filter the results to only include those where avg_pearson_coefficients > 0.9 and rmse < 0.05
        filtered_results = [result for result in results if result[1] > 0.9 and result[2] < 0.05]

        # If there are no results that meet the criteria, print a message and exit
        if filtered_results:
            # Find the best coverage and area among the filtered results
            # best_coverage = max(filtered_results, key=lambda x: x[4])
            # best_area = max(filtered_results, key=lambda x: x[5])

            for result in filtered_results:
                total_loss, avg_pearson_coefficients, rmse, coverage1, area, params = result
                log_data.append((params, total_loss, avg_pearson_coefficients, rmse, coverage1, area))

            # Find the best result
            best_result = min(log_data, key=lambda x: x[1])
            best_coverage = max(log_data, key=lambda x: x[4])
            best_area = max(log_data, key=lambda x: x[5])

            # Print the best parameter combination and loss
            print("Best Parameter Combination:", best_result[0])
            print("Best Loss:", best_result[1])

            # Print the best parameter combination and loss
            print("Best Coverage Parameter Combination:", best_coverage[0])
            print("Best Coverage:", best_coverage[4])

            # Print the best parameter combination and loss
            print("Best Area Parameter Combination:", best_area[0])
            print("Best Area:", best_area[5])

            # Dump log_data to a CSV file
            with open('src/log_data_morestuff_2.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Parameter Combination', 'Loss'])
                writer.writerows([(str(params), total_loss, avg_pearson_coefficients, rmse, coverage1, area) for params, total_loss, avg_pearson_coefficients, rmse, coverage1, area in log_data])

        else:
            print("No results meet the criteria.")

    # # Find the best result
    # best_result = min(results, key=lambda x: x[0])

    # # Print the best parameter combination and loss
    # print("Best Parameter Combination:", best_result[1])
    # print("Best Loss:", best_result[0])


#     # Update best parameters and loss if current combination is better
#     if total_loss < best_loss:
#         best_params = params
#         best_loss = total_loss


# # Print the best parameter combination and loss
# print("Best Parameter Combination:", best_params)
# print("Best Loss:", best_loss)


# ### 2.4. Hyperparameter's tuning process 

'''
YOUR CODE HERE

'''

# ## 3. Sample generation of reconstructed X0/X1 pairs and saving results 
# 
# For submission **we recommend keep num_samples equals 30** for each given Y, in any cases for the Living scoring process will be used just 30 first samples. We don't limit how many given_Y you should use. The Live Scroing will be calculated as average of them. 

# num_samples = 30 

# x_batch, y_batch = next(iter(dataloader))

# #for one random chosen one given_y
# y_idx = np.random.randint(batch_size)

# given_y = y_batch[y_idx].unsqueeze(0).to(device)

# x_output, y_output = CVAE_fn.generate_samples(model, num_samples, given_y, input_shape, device)

# CVAE_fn.plot_samples_stacked(x_output.cpu(), y_output.cpu())


#If you use several given_Y be sure that you concatanated generated samples x_output as well as y_output, i.e. next: 
    
'''
x_outputs = torch.cat([...], dim=0)   
y_outputs = torch.cat([...], dim=0) 
'''


# Save result as TensorDataset before submitting

# ds = data_utils.TensorDataset(x_output,y_output)
# torch.save(ds, 'result/result.pt') 

#Save tuned Hyperparameters 

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

# with open('result/best_hyperparameters.yaml', 'w') as f:
#     yaml.dump(hyperparameters, f)
 

# ## Submission and Evaluation
# 
# ### Submission for LiveScoring 
# 
# Upload the results as **.pt** file with x_outputs/y_outputs generated by NN on http://xeek.ai/ to score your solution and update the leaderboard. 
# 
# The evaluation of the results will combine in a total score (highlighted on the LeaderBoarded) the following assessments:
# 
#  - Reconstruction error: This measures how close the vector Y_hat obtained from the generated X0_hat and X1_hat is to the given Y. A low reconstruction error indicates that the CVAE has learned to generate combinations of X0 and X1 that can produce similar signals Y. 
#  
#  - Preservation of the structure: This measures how well the vectors X0_hat and X1_hat have retained the given structure, which is straight lines with a preserved sign of slope. A high level of preservation indicates that the CVAE has learned to generate combinations of X0 and X1 that adhere to the given structure.
# 
#  - Diversity of generated combinations: This measures how well the CVAE has covered the solution space by generating diverse combinations of X0_hat and X1_hat that can produce similar signals Y. A high level of diversity indicates that the CVAE has generated a wide range of possible combinations of X0 and X1.
# 
# Please note that a **higher total score indicates a better solution**.
# 
# ### Final Submission
# 
# Finalists will be invited to submit full solution as **ZIP** file for final review and scoring. 
# 
# This ZIP should include:
#  - **.pt file** with x_outputs/y_outputs generated by NN.  
#  - **best_hyperparameters.yaml** with hyperparameters tuned 
#  - **model_parameters.yaml** as well as 
#  - **CVAE_architecture.py** if any CVAE architecture changes were performed
# 

# ## References
# 
# W. Anderson McAliley and Yaoguo Li, (2021), "Machine learning inversion of geophysical data by a conditional variational autoencoder," SEG Technical Program Expanded Abstracts : 1460-1464.




