#!/usr/bin/env python3

# Author: Mehmed Kerem Uludag (muludag)
# Email: muludag@umich.edu
# Institution: University of Michigan
# Department: Computer Science & Engineering and Robotics
# Date: July 28, 2023
# Work: This work was started in the ADWISE Lab at Florida International University as part of the Research Expereince for Undergraduates program in the Summer of 2023. 
#
# Description: Generative Adverserial Network (GAN) implementation to to generate samples in intervals of real time series drone sensor data to be evaluated against classifiers


import sys
from pathlib import Path
from time import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

import torch
from torch import nn

from data import (extract_drone_data, extract_3multiple)
from GAN_models import (Discriminator, Generator)
from utility import (clear_line, clear_patch)
import pandas as pd
from scipy import stats
import pdb

#SEED = 13
#np.random.seed(SEED)
#torch.random.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)


def main(argv=[]): 
    # model params
    N = 1500 #has to be divisble by minibatch size
    num_epochs: int = 400
    minibatch_size: int = 25
    d_learning_rate: float = 0.0001
    g_learning_rate: float = 0.0001
    discriminator_optim: str = 'adam'
    generator_optim: str = 'sgd'
    loss_type: str = 'mse'
    z_dim: int = 5
    data_dim: int = 1
    d_hidden_size: int = 30
    g_hidden_size: int = 30
    progress_update_interval = 20
    save_figs: bool = False
    save_model: bool = False

    experiment_info = f'\ntotal iters: {num_epochs},\nbatch_size: {minibatch_size},\nd_lr: {d_learning_rate},\n' + \
                      f'g_lr: {g_learning_rate},\nloss: {loss_type},\nd_hidden_size: {d_hidden_size},\n' + \
                      f'g_hidden_size: {g_hidden_size},\ndisc_optim: {discriminator_optim},\n' + \
                      f'gen_optim: {generator_optim},\ndata_dim: {data_dim},\nz_dim: {z_dim},\n' + \
                      f'random seed: '

    # input data
    datafile = 'ace-benign-log_0_2033-8-19-16-27-30_sensor_combined_0'
    column_names = ['gyro_rad[0]']#, 'gyro_rad[1]', 'gyro_rad[2]']#, 'accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]']
    outfilename = 'delete.csv'
    #data_dim = 1
    #train_labels = torch.zeros((N, data_dim))
    _data = extract_drone_data(N, column_names, datafile, torch.device('cpu')).numpy()
    train_data = torch.from_numpy(_data).float()
    #train_set = [(train_data[i], train_labels[i]) for i in range(N)]
    fig, ax_data, ax_loss, ax_disc = prepare_plots2(_data, experiment_info, '1D Gaussian')

    # switch to gpu for the hell of it
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    # build discrimniator and Generator (Test out custom ones in the future)
    D = Discriminator(data_dim, d_hidden_size).to(device)
    G = Generator(z_dim, g_hidden_size, data_dim).to(device)

    # optimizers for D and G (Adam)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))

    # Arrays to be used in training 
    show_separate_loss = False
    d_real_loss_list: list = []
    d_fake_loss_list: list = []
    g_loss_list: list = []
    d_x_list: list = []
    d_g_z_list: list = []

    # setup for training data
    K = minibatch_size # window height (by data_dim) # has to be equal to batch size for model
    window_data = []
    stride = K
    for i in range(0, len(train_data) - K + 1, stride):
        window = train_data[i:i+K]
        window_data.append(window)
    
    # Create DataLoader objects for each window
    # window_loaders = []
    # for window in window_data:
    #     window_loader = torch.utils.data.DataLoader(window, batch_size=minibatch_size, shuffle=False)
    #     window_loaders.append(window_loader)
    
    
    # define loss function (test different loss functions, eventually implement custom one, Quantile Loss, Time Series Cross Entropy)
    
    def handle_outliers(generated_samples_window):
        # Calculate mean and standard deviation of the generated samples
        window_mean = np.mean(generated_samples_window)
        window_std = np.std(generated_samples_window)
        print(window_mean, " ", window_std)
        # Define the threshold as a range between -3 and 3 standard deviations
        lower_threshold = window_mean - 1 * window_std
        upper_threshold = window_mean + 1 * window_std

        print(lower_threshold, " ", upper_threshold)
        # Replace outliers with values within the threshold range
        for i, sample in enumerate(generated_samples_window):
            if sample < lower_threshold:
                generated_samples_window[i] = lower_threshold
            elif sample > upper_threshold:
                generated_samples_window[i] = upper_threshold

        return generated_samples_window
    
    
    loss_function = nn.SmoothL1Loss()
    
    def apply_smoothening(data: torch.Tensor, smoothing_factor: float) -> torch.Tensor:
        smoothed_data = [data[0]]
        for i in range(1, len(data)):
            smoothed_value = (1 - smoothing_factor) * smoothed_data[-1] + smoothing_factor * data[i]
            smoothed_data.append(smoothed_value)
        return torch.tensor(smoothed_data)


    generated_samples_all_windows = []
    
    # Training
    t1 = time()
    max_iter = num_epochs# * len(window_data) // 10 # Maximum number of iterations
    
    for window in window_data:
        window_loader = torch.utils.data.DataLoader(window, batch_size=minibatch_size, shuffle=False)
        for iteration in range(1, max_iter + 1):
            for n, samples in enumerate(window_loader):
                if isinstance(samples, tuple):
                    real_samples, _ = samples
                else:
                    real_samples = samples                
                
                # Data for training the discriminator
                real_samples_labels = torch.ones((minibatch_size, 1))
                latent_space_samples = torch.randn((minibatch_size, z_dim))
                generated_samples = G(latent_space_samples)
                generated_samples_labels = torch.zeros((minibatch_size, 1))
                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

                # Train Discriminator: 
                D.zero_grad() # thid needed? 
                output_discriminator = D(all_samples)
                #pdb.set_trace()
                loss_discriminator = loss_function(output_discriminator[0], all_samples_labels)
                loss_discriminator.backward()
                d_optimizer.step()

                # Train Generator: 
                latent_space_samples = torch.randn((minibatch_size, z_dim))
                G.zero_grad()
                generated_samples = G(latent_space_samples)
                # Apply smoothening
                smoothing_factor = 0.01
                #pdb.set_trace()
                generated_samples_s = apply_smoothening(generated_samples, smoothing_factor).reshape(K, data_dim) #Could be just 1 instead of DATA_DIMSSSSSSSSSSSSSSSSS 
                output_discriminator_generated = D(generated_samples)
                loss_generator = loss_function(output_discriminator_generated[0], real_samples_labels)
                loss_generator.backward()
                g_optimizer.step()

            # Print losses:
            if iteration % progress_update_interval == 0 or iteration == num_epochs:
                #print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                #print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                print(
                        f'iteration: {iteration} '
                        f'generator_loss: {loss_generator} '
                        f'discriminator_loss: {loss_discriminator}')
                
                g_loss_list.append(loss_generator.tolist())
                if 1 == 0:
                    # plot losses
                    update_loss_plot(
                        ax_loss, d_real_loss_list, d_fake_loss_list, g_loss_list,
                        progress_update_interval, show_separate_loss)

                    # plot d(x) and d(g(z))
                    d_x_list.append(torch.mean(D(real_samples)).item())
                    d_g_z_list.append(torch.mean(G(latent_space_samples)).item())
                    update_disc_plot(ax_disc, d_x_list, d_g_z_list, progress_update_interval)

                    # plot generated data
                    z = torch.randn(N, z_dim)
                    fake_data = G(z).detach().cpu().numpy()
                    update_data_plot(ax_data, D, fake_data, device)

                    # Refresh figure
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    
        
        z = torch.randn(K, z_dim)
        generated_samples_window = G(z).detach().cpu().numpy()
        generated_samples_window = handle_outliers(generated_samples_window)
        generated_samples_all_windows.append(generated_samples_window)


    x = np.arange(0, N)
    generated_samples = np.concatenate(generated_samples_all_windows, axis=0)

    plt.scatter(x, train_data, label='Real Data', marker='.')
    plt.scatter(x, generated_samples, label='Generated Samples', marker='.')
    plt.legend()
    plt.savefig("gan_plots/results.png")
    plt.show()
    plt.clf()
    
    #save the generated samples: 
    df = pd.DataFrame(data=generated_samples.flatten(), columns=column_names)
    df.to_csv(outfilename, index=False)
    

    window_size = 10
    train_column_data = train_data.flatten().numpy()
    train_moving_avg = np.convolve(train_column_data, np.ones(window_size) / window_size, mode='same')
    train_z_scores = stats.zscore(train_column_data)
    z_score_threshold = 2.5
    train_outliers = np.where(np.abs(train_z_scores) > z_score_threshold)[0]

    generated_column_data = generated_samples.flatten()
    generated_moving_avg = np.convolve(generated_column_data, np.ones(window_size) / window_size, mode='same')
    generated_z_scores = stats.zscore(generated_column_data)
    generated_outliers = np.where(np.abs(generated_z_scores) > z_score_threshold)[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Plot train_data
    ax1.plot(train_column_data, label='Original Data')
    ax1.plot(train_moving_avg, label='Moving Average')
    ax1.scatter(train_outliers, train_column_data[train_outliers], color='red', label='Outliers')
    ax1.set_ylabel('Value')
    ax1.legend()

    # Plot generated_samples
    ax2.plot(generated_column_data, label='Generated Data')
    ax2.plot(generated_moving_avg, label='Moving Average')
    ax2.scatter(generated_outliers, generated_column_data[generated_outliers], color='red', label='Outliers')
    ax2.set_xlabel('Data Point')
    ax2.set_ylabel('Value')
    ax2.legend()

    plt.savefig("gan_plots/train_and_generated_data")
    plt.show()

    plt.clf()

    return



def prepare_plots2(data, info, title=''):
    fig = plt.figure(1, figsize=(14, 8.0))
    fig.canvas.setWindowTitle(title)

    ax_data = fig.add_subplot(3, 1, 1)
    ax_loss = fig.add_subplot(3, 1, 2)
    ax_disc = fig.add_subplot(3, 1, 3)

    fig.tight_layout(h_pad=1.55, rect=[0.01, 0.04, 0.99, 0.98])

    ax_data.set_title(title, fontweight='bold')
    ax_data.plot(data, label='data', color='royalblue', marker='.', markersize=7)
    

    ax_data.annotate(
        info.replace('\n', '  '),
        xy=(0, 0),
        xytext=(2, 14),
        xycoords=('figure pixels', 'figure pixels'),
        textcoords='offset pixels',
        bbox=dict(facecolor='dodgerblue', alpha=0.15),
        size=9.5,
        ha='left'
    )

    #ax_data.set_ylim(bottom=-10, top=120)

    ax_loss.set_title('Losses', fontweight='bold')
    ax_loss.grid()

    ax_disc.set_title('Discriminator Outputs', fontweight='bold')
    ax_disc.grid()

    return fig, ax_data, ax_loss, ax_disc



def update_loss_plot(ax: plt.Axes, d_loss, d_fake_loss, g_loss, update_interval, separate=False):
    clear_line(ax, 'd_loss')
    clear_line(ax, 'g_loss')

    x = np.arange(1, len(d_loss) + 1)

    if separate:
        ax.plot(x, np.add(d_loss, d_fake_loss), color='dodgerblue', label='D Loss', gid='d_loss')
        clear_line(ax, 'd_real_loss')
        ax.plot(x, d_loss, color='lightseagreen', label='D Loss(Real)', gid='d_real_loss')
        clear_line(ax, 'd_fake_loss')
        ax.plot(x, d_fake_loss, color='mediumpurple', label='D Loss(Fake)', gid='d_fake_loss')
    else:
        ax.plot(x, d_loss, color='dodgerblue', label='D Loss', gid='d_loss')

    ax.plot(x, g_loss, color='coral', label='G Loss', gid='g_loss', alpha=0.9)
    ax.legend(loc='upper right', framealpha=0.75)
    ax.set_xlim(left=1, right=len(x) + 0.01)
    ticks = ax.get_xticks()
    ax.set_xticklabels([f'{t:.0f}' for t in ticks * update_interval])


def update_disc_plot(ax: plt.Axes, d_x, d_g_z, update_interval):
    clear_line(ax, 'dx')
    clear_line(ax, 'dgz')

    x = np.arange(1, len(d_x) + 1)
    ax.plot(x, d_x, color='#308862', label='D(x)', gid='dx')
    ax.plot(x, d_g_z, color='#B23F62', label='D(G(z))', gid='dgz', alpha=0.9)
    ax.legend(loc='upper right', framealpha=0.75)
    ax.set_xlim(left=1, right=len(x) + 0.01)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1, 0.1))
    ticks = ax.get_xticks()
    ax.set_xticklabels([f'{t:.0f}' for t in ticks * update_interval])


def update_data_plot(ax, D, fake_data, device):
    # draw decision discriminator boundary
    clear_line(ax, 'decision')
    plot_decision_boundary(ax, D, device)
    #
    clear_patch(ax, 'g_hist')
    ax.hist(
        fake_data,
        gid='g_hist',
        bins=100,
        density=True,
        histtype='stepfilled',
        label='generated',
        facecolor='moccasin',
        edgecolor='sandybrown',
        linewidth=2,
        alpha= 0.85
    )
    ax.legend(loc='upper right', framealpha=0.75)


def plot_decision_boundary(ax: plt.Axes, discriminator, device=torch.device('cpu')) -> None:
    _data = torch.linspace(-5, 9, 3000, requires_grad=False).view(3000, 1).to(device)
    decision = discriminator(_data)
    if type(decision) == tuple:
        decision = decision[0]

    ax.plot(
        _data.cpu().numpy(),
        decision.detach().cpu().numpy(),
        gid='decision',
        label='decision boundary',
        color='gray',
        linestyle='--'
    )

if __name__ == "__main__":
    main(sys.argv)
