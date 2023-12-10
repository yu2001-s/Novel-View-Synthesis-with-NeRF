# Script for training the NeRF model.


import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from src.utils import *
from src.data_loader import *
from src.model import *
from src.trainer import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
seed = 42
torch.manual_seed(seed)

# Set hyperparameters
N_samples = 64

