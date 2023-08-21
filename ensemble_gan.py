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
from GAN_models import (Multi_Discriminator, Multi_Generator, UAV_State_Discriminator)
from utility import (clear_line, clear_patch)
import pandas as pd
from scipy import stats
import pdb


def main(argv=[]):
    num_epochs: int = 40
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
    
    num_states = 3  # Take-off, hovering, landing
    # Calculate number of samples for each state based on total number of data points
    num_samples_takeoff = int(N * 0.1)
    num_samples_normal = int(N * 0.8)
    num_samples_landing = int(N * 0.1)
    
    # Create a list of state labels based on the calculated counts
    state_labels = [0] * num_samples_takeoff + [1] * num_samples_normal + [2] * num_samples_landing 
    
    D = Multi_Discriminator(data_dim, d_hidden_size).to(device)
    G = Multi_Generator(z_dim, g_hidden_size, data_dim).to(device)
    S = UAV_State_Discriminator(data_dim, d_hidden_size, num_states).to(device)
    
    d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))
    state_discriminator_optim = torch.optim.Adam(S.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
    
    K = minibatch_size # window height (by data_dim) # has to be equal to batch size for model
    window_data = []
    stride = K
    for i in range(0, len(train_data) - K + 1, stride):
        window = train_data[i:i+K]
        window_data.append(window)
    
    num_windows_takeoff = int(len(window_data) * 0.1)
    num_windows_normal = int(len(window_data) * 0.8)
    num_windows_landing = int(len(window_data) * 0.1)
    
    window_data_takeoff = window_data[:num_windows_takeoff]
    window_data_normal = window_data[num_windows_takeoff:num_windows_takeoff + num_windows_normal]
    window_data_landing = window_data[-num_windows_landing:]

    state_labels_takeoff = [0] * num_windows_takeoff
    state_labels_normal = [1] * num_windows_normal
    state_labels_landing = [2] * num_windows_landing

    window_data_categories = window_data_takeoff + window_data_normal + window_data_landing
    state_labels_categories = state_labels_takeoff + state_labels_normal + state_labels_landing
    
    loss_function = nn.SmoothL1Loss()
    generated_samples_all_windows = []
    #generated_samples_all_windows = np.zeros_like(train_data)
    
    lambda_state = 0.1  # Adjust this value based on experimentation
    
    # Training
    for window, state_label in zip(window_data_categories, state_labels_categories):
        window_loader = torch.utils.data.DataLoader(window, batch_size=minibatch_size, shuffle=False)
        for epochs in range(1, num_epochs + 1):
            for n, samples in enumerate(window_loader):
                if isinstance(samples, tuple):
                    real_samples, _ = samples
                else:
                    real_samples = samples     
                               
                # Train State Discriminator:
                S.zero_grad()
                output_state = S(real_samples.view(-1, data_dim))
                state_labels_batch = torch.tensor([[state_label]] * minibatch_size)  # shape (minibatch_size, 1)  

                loss_state = loss_function(output_state, state_labels_batch)
                loss_state.backward()
                state_discriminator_optim.step()
                
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
                #generated_samples = G(torch.cat((latent_space_samples, state_label), dim=1)) #can customize G to generate based on current state
                output_discriminator_generated = D(generated_samples)
                # loss_generator = loss_function(output_discriminator_generated[0], real_samples_labels) #old
                # imporved training: use state discriminator's output to provide feedback to the generator 
                loss_state_feedback = loss_function(output_discriminator_generated[0], state_labels_batch)
                loss_generator = loss_function(output_discriminator_generated[0], real_samples_labels) + lambda_state * loss_state_feedback
                loss_generator.backward()
                g_optimizer.step()

            # Print losses:
            if epochs % progress_update_interval == 0 or epochs == num_epochs:
                #print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                #print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                print(
                    f'iteration: {epochs} '
                    f'state_loss: {loss_state} '
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
    
    
    
    # Evaluating Generated Data:
    # This G should be just on last window data, so like landing data. 
    num_fake_samples = 100
    z = torch.randn(num_fake_samples, z_dim)
    generated_samples = G(z)

    # Pass the generated samples through the state discriminator
    output_state_generated = S(generated_samples.view(-1, data_dim))

    # Calculate the average probabilities for each state
    average_probs = output_state_generated.mean(dim=0)

    print("Average probabilities for each state:")
    print("Take-off:", average_probs[0].item())
    print("Normal flight:", average_probs[1].item())
    print("Landing:", average_probs[2].item())
    
    return












if __name__ == "__main__":
    main(sys.argv)