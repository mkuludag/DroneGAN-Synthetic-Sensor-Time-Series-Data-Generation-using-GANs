#!/usr/bin/env python3

# Author: Mehmed Kerem Uludag (muludag)
# Email: muludag@umich.edu
# Institution: University of Michigan
# Department: Computer Science & Engineering and Robotics
# Date: July 28, 2023
# Work: This work was started in the ADWISE Lab at Florida International University as part of the Research Expereince for Undergraduates program in the Summer of 2023. 
# 
# Description: Generator Nerual Network Architecture and description for GAN


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor


class Generator(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.data_dim = data_dim

        self.l1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ReLU()
        )
        self.conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, data_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.l1(x)
        out = out.unsqueeze(-1)  # Add a dimension for the channel
        out = self.conv1d(out)
        out = out.squeeze(-1)  # Remove the channel dimension
        out = out.unsqueeze(1)  # Reshape for LSTM input
        out, _ = self.lstm(out)
        out = out[:, -1, :]  # Take the last timestep output
        out = self.fc(out)
        return out
    
    
class Multi_Generator(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super(Multi_Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.data_dim = data_dim

        self.l1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ReLU()
        )
        self.conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, data_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should be of size (k, M)
        out = self.l1(x)
        out = self.conv1d(out.transpose(1, 2))  # Transpose  for Conv1d
        out, _ = self.lstm(out)
        out = out[:, -1, :]  # Take the last timestep output
        out = self.fc(out)
        return out


    
# original
# class Generator(nn.Module):
#     def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
#         super().__init__()
#         self.l1 = nn.Sequential(
#             nn.Linear(z_dim, hidden_size),
#             #nn.ELU(alpha=0.2)
#             nn.ReLU()
#         )
#         self.l2 = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             #nn.ELU(alpha=0.2)
#             nn.ReLU()
#         )
#         self.l3 = nn.Linear(hidden_size, data_dim)

#     def forward(self, x: Tensor) -> Tensor:
#         out = self.l1(x)
#         out = self.l2(out)
#         out = self.l3(out)

#         return out


class TimeSeriesGenerator(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ELU(alpha=0.2)
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
        )
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)

        return out


class Generator8G(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ELU(alpha=0.2)
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
        )
        self.l3 = nn.Linear(hidden_size, data_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        
        return out


class Generator25G(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l4 = nn.Linear(hidden_size, data_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
    
        return out


class GeneratorMixture(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(z_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size * 2)
        self.l3 = nn.Linear(hidden_size * 2, hidden_size)
        self.l4 = nn.Linear(hidden_size, data_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))

        return self.l4(out)


class GeneratorMNIST(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ELU(alpha=0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc1[0].out_features, self.fc1[0].out_features * 2),
            nn.ELU(alpha=0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.fc2[0].out_features, self.fc2[0].out_features * 2),
            nn.ELU(alpha=0.2)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(self.fc2[0].out_features * 2, data_dim),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
    
        return out