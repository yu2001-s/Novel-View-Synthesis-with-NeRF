import torch
from tqdm import tqdm
import wandb
from datetime import datetime
from src.utils import *

class NeRFTrainer:
    def __init__(self, model, optimizer, lr_scheduler, loss_fn, train_loader, val_loader, device, wandb_run=None, obj_name="chair"):
        """
        NeRF trainer with validation, checkpoint saving, early stopping, visualization, and Wandb logging.

        Args:
            model: NeRF model
            optimizer: Optimizer for training
            lr_scheduler: Learning rate scheduler
            loss_fn: Loss function
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training (CPU or CUDA)
            wandb_run: Weights & Biases run (optional)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wandb_run = wandb_run
        self.obj_name = obj_name

    def train_step(self, x, d, z_vals, target_rgb):
        """
        Perform a training step with loss calculation and gradient update.

        Args:
            x: Input positions
            d: Input view directions
            target_rgb: Target RGB color

        Returns:
            Training loss
        """
        try:
            # Forward pass
            rgb, sigma = self.model(x, d)
            z_vals = z_vals.squeeze(-1)
            rendered_rgb = volume_rendering(z_vals, rgb, sigma, white_bkgd=False)
            # Calculate loss
            loss = self.loss_fn(rendered_rgb, target_rgb)

            # Backward pass and update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()
        except Exception as e:
            print(f"Error during training: {e}")
            raise e

    def validate(self):
        """
        Evaluate the model on the validation set and return the average loss.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in self.val_loader:
                # Unpack the data from the dataset
                points = data['points'].to(self.device)
                v_dir = data['v_dir'].to(self.device)
                target_rgb = data['rgb'].to(self.device)
                z_vals = data['z_vals'].to(self.device).squeeze(-1)  # Ensure z_vals are provided by the dataset

                # Forward pass through the model
                rgb, sigma = self.model(points, v_dir)

                # Perform volume rendering using the outputs from the model
                rendered_rgb = volume_rendering(z_vals, rgb, sigma, white_bkgd=False)

                # Calculate the loss using the rendered RGB and the target RGB
                loss = self.loss_fn(rendered_rgb, target_rgb)
                total_loss += loss.item()

        # Clearing the CUDA cache after validation can help with memory management,
        # but might not be necessary. Use it if you face memory issues.
        torch.cuda.empty_cache()

        average_loss = total_loss / len(self.val_loader)
        return average_loss

    def train(self, epochs, log_interval=100, early_stopping_patience=5):
        """
        Train the model for a specified number of epochs with validation, checkpoint saving, early stopping, and logging.

        Args:
            epochs: Number of epochs to train
            log_interval: Frequency of printing training and validation losses
            early_stopping_patience: Number of epochs without validation loss improvement before stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            # Training loop
            self.model.train()
            total_train_loss = 0
            tbar = tqdm(enumerate(self.train_loader))
            for batch_idx, data in tbar:
                # Unpack the data from the dataset
                x = data['points'].to(self.device)
                d = data['v_dir'].to(self.device)
                z_vals = data['z_vals'].to(self.device) 
                target_rgb = data['rgb'].to(self.device)

                # Perform a training step
                loss = self.train_step(x, d, z_vals, target_rgb)
                total_train_loss += loss
                tbar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

            # Validation and early stopping
            val_loss = self.validate()
            if self.wandb_run:
                wandb.log({"epoch": epoch, "train_loss": total_train_loss / len(self.train_loader), "val_loss": val_loss})

            if val_loss < best_val_loss:
                patience_counter = 0
                best_val_loss = val_loss
                # Save model checkpoint with timestamp
                torch.save(self.model.state_dict(), f"models/{self.obj_name}_epoch_{epoch+1}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Update learning rate and log progress
            self.lr_scheduler.step()
            if epoch % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_train_loss / len(self.train_loader):.4f} | Val Loss: {val_loss:.4f}")