#!/usr/bin/env python3

# Author: Mehmed Kerem Uludag (muludag)
# Email: muludag@umich.edu
# Institution: University of Michigan
# Department: Computer Science & Engineering and Robotics
# Date: July 28, 2023
# Work: This work was started in the ADWISE Lab at Florida International University as part of the Research Expereince for Undergraduates program in the Summer of 2023. 
#
# Description: Discriminator Nerual Network Architecture and description for GAN


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor





class Discriminator(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(data_dim, hidden_size),
            #nn.ELU(alpha=0.05)
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            #nn.ELU(alpha=0.05)
            nn.ReLU()
        )
        self.l3 = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        feat = self.l2(out)
        out = self.l3(feat)
        out = self.sigmoid(out)

        return out, feat
    

class Multi_Discriminator(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(data_dim, hidden_size),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.l3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should be of size (k, M)
        out = self.l1(x)
        feat = self.l2(out)
        out = self.l3(feat)
        out = self.sigmoid(out)
        
        return out, feat


# class Discriminator(nn.Module):
#     def __init__(self, data_dim: int, hidden_size: int) -> None:
#         super(Discriminator, self).__init__()
#         self.data_dim = data_dim
#         self.hidden_size = hidden_size

#         self.conv1d = nn.Conv1d(data_dim, hidden_size, kernel_size=3, padding=1)
#         self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         out = x.unsqueeze(1)  # Add a channel dimension
#         out = self.conv1d(out)
#         out = out.squeeze(-1)  # Remove the channel dimension
#         out, _ = self.lstm(out)
#         out = out[:, -1]  # Take the last timestep output
#         #out = self.fc(out)
#         out = self.sigmoid(out)

#         return out, out
    
    
# class Discriminator(nn.Module):
#     def __init__(self, data_dim: int, hidden_size: int) -> None:
#         super().__init__()
#         self.l1 = nn.Sequential(
#             nn.Linear(data_dim, hidden_size),
#             nn.ReLU()
#         )
#         self.l2 = nn.Sequential(
#             nn.Conv1d(data_dim, hidden_size, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.l3 = torch.nn.Linear(hidden_size, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x: Tensor) -> Tensor:
#         out = self.l1(x)
#         out = x.unsqueeze(1)
#         out = self.l2(out)
#         out = out.squeeze(-1)
#         out = self.l3(out)
#         out = self.sigmoid(out)

#         return out

class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int) -> None:
        super(TimeSeriesDiscriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(data_dim, hidden_size),
            nn.ELU(alpha=0.05)
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.05)
        )
        self.l3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = feat = self.l2(out)
        out = out.view(out.size(0), -1)
        out = self.l3(out)
        out = self.sigmoid(out)

        return out, feat





class Discriminator8G(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(data_dim, hidden_size),
            nn.ELU(alpha=0.05)
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(alpha=0.05)
        )
        self.l3 = torch.nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.sigmoid(out)
        
        return out


class Discriminator25G(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(data_dim, hidden_size),
            nn.ELU(alpha=0.05)
            # nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.05)
            # nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(alpha=0.05)
            # nn.ReLU()
        )
        self.l4 = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.sigmoid(out)
    
        return out


class DiscriminatorMixture(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(data_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size * 2)
        self.l3 = nn.Linear(hidden_size * 2, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = torch.sigmoid(self.l4(out))

        return out


class DiscriminatorMNIST(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(data_dim, hidden_size),
            nn.ELU(alpha=0.2),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc1[0].out_features, self.fc1[0].out_features // 2),
            nn.ELU(alpha=0.2),
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.fc2[0].out_features, self.fc2[0].out_features // 2),
            nn.ELU(alpha=0.2),
            nn.Dropout(0.3)
        )
        self.fc4 = nn.Linear(self.fc3[0].out_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)

        return out
