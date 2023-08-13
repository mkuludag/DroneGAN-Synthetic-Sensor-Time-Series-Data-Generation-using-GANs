#!/usr/bin/env python3

# Author: Mehmed Kerem Uludag (muludag)
# Email: muludag@umich.edu
# Institution: University of Michigan
# Department: Computer Science & Engineering and Robotics
# Date: July 28, 2023
# Work: This work was started in the ADWISE Lab at Florida International University as part of the Research Expereince for Undergraduates program in the Summer of 2023. 
#
# Description: Ensemble GAN

import sys
from pathlib import Path
from time import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

import torch
from torch import nn

from data import (extract_3multiple)
from GAN_models import (Multi_Discriminator, Multi_Generator)
from utility import (clear_line, clear_patch)
import pandas as pd
from scipy import stats
import pdb


def main(argv=[]):
    num_epochs: int = 400
    minibatch_size: int = 25
    d_learning_rate: float = 0.0001
    g_learning_rate: float = 0.0001
    discriminator_optim: str = 'adam'
    generator_optim: str = 'sgd'
    loss_type: str = 'mse'
    z_dim: int = 5
    data_dim: int = 3
    d_hidden_size: int = 30
    g_hidden_size: int = 30
    progress_update_interval = 20
    device = torch.device('cpu')
    
    # Input Data:    
    outfilename = 'ensemble_out.csv'    
    d1 = 'gyro0_gen'
    d2 = 'hover_thrust_gen1'
    d3 = 'xyz_gen0' 
    
    N = 1000
    data = extract_3multiple(N, d1, d2, d3, torch.device('cpu')).numpy()
    train_data = torch.from_numpy(data).float()
    M = train_data.shape[1]
    
    D = Multi_Discriminator(data_dim, d_hidden_size).to(device)
    G = Multi_Generator(z_dim, g_hidden_size, data_dim).to(device)
    
    d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))
    
    K = minibatch_size # window height (by data_dim) # has to be equal to batch size for model
    window_data = []
    stride = K
    for i in range(0, len(train_data) - K + 1, stride):
        window = train_data[i:i+K]
        window_data.append(window)
    
    loss_function = nn.SmoothL1Loss()
    generated_samples_all_windows = []
    #generated_samples_all_windows = np.zeros_like(train_data)
    
    # Training
    for window in window_data:
        window_loader = torch.utils.data.DataLoader(window, batch_size=minibatch_size, shuffle=False)
        for epochs in range(1, num_epochs + 1):
            for n, samples in enumerate(window_loader):
                if isinstance(samples, tuple):
                    real_samples, _ = samples
                else:
                    real_samples = samples                
                
                # Data for training the discriminator
                real_samples_labels = torch.ones((minibatch_size, 1))
                latent_space_samples = torch.randn((minibatch_size, z_dim))
                #pdb.set_trace()
                generated_samples = G(latent_space_samples)
                generated_samples_labels = torch.zeros((minibatch_size, 1))
                #print(real_samples.shape, " ", generated_samples.shape)
                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

                # Train Discriminator: 
                D.zero_grad() # thid needed? 
                output_discriminator = D(all_samples)
                loss_discriminator = loss_function(output_discriminator[0], all_samples_labels)
                loss_discriminator.backward()
                d_optimizer.step()

                # Train Generator: 
                latent_space_samples = torch.randn((minibatch_size, z_dim))
                G.zero_grad()
                generated_samples = G(latent_space_samples)
                output_discriminator_generated = D(generated_samples)
                loss_generator = loss_function(output_discriminator_generated[0], real_samples_labels)
                loss_generator.backward()
                g_optimizer.step()

            # Print losses:
            if epochs % progress_update_interval == 0 or epochs == num_epochs:
                #print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                #print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                print(
                        f'iteration: {epochs} '
                        f'generator_loss: {loss_generator} '
                        f'discriminator_loss: {loss_discriminator}')                   
        
        z = torch.randn(K, z_dim)
        generated_samples_window = G(z).detach().cpu().numpy()
        generated_samples_all_windows.append(generated_samples_window)
    
    generated_samples_all_windows = np.array(generated_samples_all_windows)
    
    # Create a figure with M subplots
    G_reshaped = generated_samples_all_windows.reshape(train_data.shape)
    x = np.arange(0, N)
    fig, axes = plt.subplots(nrows=M, ncols=1, figsize=(8, 6 * M))
    for i in range(M):
        ax = axes[i]
        ax.scatter(x, train_data[:, i], label='Train Data', color='blue')
        ax.scatter(x, G_reshaped[:, i], label='Generated Data', color='orange')
        ax.set_xlabel('Time Stamps')
        ax.set_ylabel('Data Sensor Readings')
        ax.set_title(f'Graph {i+1}')
        ax.legend()

    plt.tight_layout()

    # Show the figure
    plt.savefig("ensemble_gan_plots/all_sensors.png")
    plt.show()
    
    return












if __name__ == "__main__":
    main(sys.argv)