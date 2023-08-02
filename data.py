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

