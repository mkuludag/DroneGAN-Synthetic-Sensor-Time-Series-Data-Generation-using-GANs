#!/usr/bin/env python3

# Author: Mehmed Kerem Uludag (muludag)
# Email: muludag@umich.edu
# Institution: University of Michigan
# Department: Computer Science & Engineering and Robotics
# Date: July 28, 2023
# Work: This work was started in the ADWISE Lab at Florida International University as part of the Research Expereince for Undergraduates program in the Summer of 2023. 
#
# Description: Helper Functions for Data Pre-processing

import random
import numpy as np
import torch
from torch import Tensor
from typing import List
import pandas as pd



def extract_drone_data(rows: int, column_names: list, filename: str, device: torch.device) -> Tensor:
    path =  'sensor-csvfiles/' + filename + '.csv'
    data = pd.read_csv(path)
    columns_to_extract = column_names
    S = 0
    A = rows + S
    extracted_data = data[columns_to_extract][S:A]
    arr = extracted_data.values
    
    return torch.tensor(arr,
                        dtype=torch.float,
                        requires_grad=False,
                        device=device)

def extract_3multiple(rows: int, f1: str, f2: str, f3: str, device: torch.device) -> Tensor:
    p1 = 'sensor-csvfiles/' + f1 + '.csv'
    p2 = 'sensor-csvfiles/' + f2 + '.csv'
    p3 = 'sensor-csvfiles/' + f3 + '.csv'
    
    data1 = pd.read_csv(p1)
    data2 = pd.read_csv(p2)
    data3 = pd.read_csv(p3)
    
    extracted_data1 = data1[:rows]
    extracted_data2 = data2[:rows]
    extracted_data3 = data3[:rows]
    
    # Combine the data horizontally to create a single DataFrame
    combined_data = pd.concat([extracted_data1, extracted_data2, extracted_data3], axis=1)
    arr = combined_data.values
    
    return torch.tensor(arr,
                        dtype=torch.float,
                        requires_grad=False,
                        device=device)