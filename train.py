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
SCALEDOWN = 2
OBJ_NAME = 'chair'
BATCH_SIZE = 32
NUM_WORKERS = 4
SAMPLE = 32 
D = 6
W = 128
input_ch_pos = 3
input_ch_dir = 2
L_p = 10
L_v = 4
skips = [3]
lr = 1e-3

img_size = int(800/SCALEDOWN)

# Set paths
P_PATH = os.path.join(os.getcwd())
sys.path.append(P_PATH)


# Load data
train_dataset = SynDataset(obj_name=OBJ_NAME, root_dir=P_PATH, split='train', img_size=img_size, num_points=SAMPLE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

min_max = train_dataset.min_max

val_dataset = SynDataset(obj_name=OBJ_NAME, root_dir=P_PATH, split='val', img_size=img_size, num_points=SAMPLE, min_max=min_max)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Initialize model

wandb.init(project="nerf",
              name=f"{OBJ_NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
              config={
                "D": D,
                "W": W,
                "L_p": L_p,
                "L_v": L_v,
                "lr": lr,
                "batch_size": BATCH_SIZE,
                "sample_per_ray": SAMPLE,
                "obj_name": OBJ_NAME,
                "img_size": img_size,
                "min_max": min_max
              })


model = NeRF(D=D, W=W, input_ch_pos=input_ch_pos, input_ch_dir=input_ch_dir, skips=skips, L_p=L_p, L_v=L_v).to(device)
model = model.to(device)


# Initialize optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=lr)
#set learning rate scheduler to not decay
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1.0)

# Set loss function
loss_fn = nn.MSELoss()

# Initialize trainer
trainer = NeRFTrainer(model=model, optimizer=optimizer, 
                      lr_scheduler=lr_scheduler, loss_fn=loss_fn, 
                      train_loader=train_dataloader, val_loader=val_dataloader, 
                      device=device, wandb_run=True)

# Train model
trainer.train(epochs=100, log_interval=1, early_stopping_patience=5)
wandb.finish()
