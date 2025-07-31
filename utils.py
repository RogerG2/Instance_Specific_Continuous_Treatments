import random
import numpy as np
import torch
import sys
import os

def set_seed(seed):
    # Set the seed for random number generation in Python
    random.seed(seed)

    # Set the seed for NumPy (if you use NumPy)
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

def disable_print():
    sys._stdout = sys.stdout  # Save original stdout
    sys.stdout = open(os.devnull, 'w')

# Function to enable printing
def enable_print():
    sys.stdout.close()
    sys.stdout = sys._stdout
