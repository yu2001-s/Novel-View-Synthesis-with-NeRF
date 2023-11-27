#Implementation of the NeRF model

import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(scene_5d, L_loc=10, L_dir=4):
    """
    Positional encoding for the input.

    Args:
        scene_5d (torch.Tensor): Input scene_5d. x, y, z, theta, phi
        L_loc (int, optional): Number of frequencies for the location. Defaults to 10.
        L_dir (int, optional): Number of frequencies for the direction. Defaults to 4.

    Returns:
        torch.Tensor: (shape: batch_size, L_loc * 2)
        torch.Tensor: (shape: batch_size, L_dir * 2)
    
    """
    # Separate location (x, y, z) and direction (theta, phi) from the input
    loc = scene_5d[:, :3]  # first three dimensions (x, y, z) shape: (batch_size, 3)
    dir = scene_5d[:, 3:]  # last two dimensions (theta, phi) shape: (batch_size, 2)

    # Initialize frequencies for location and direction
    freqs_loc = 2 ** torch.arange(L_loc, dtype=torch.float32)
    freqs_dir = 2 ** torch.arange(L_dir, dtype=torch.float32)

    # Apply sinusoidal encoding for location
    loc_encoded = [torch.sin(loc * f) for f in freqs_loc] + [torch.cos(loc * f) for f in freqs_loc]
    loc_encoded = torch.cat(loc_encoded, dim=-1)

    # Apply sinusoidal encoding for direction
    dir_encoded = [torch.sin(dir * f) for f in freqs_dir] + [torch.cos(dir * f) for f in freqs_dir]
    dir_encoded = torch.cat(dir_encoded, dim=-1)

    return loc_encoded, dir_encoded



class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=60, input_ch_views=16, output_ch=4):
        """
        NeRF Model.

        Args:
            D (int): Number of layers in MLP.
            W (int): Number of neurons in each layer of MLP.
            input_ch (int): Number of channels in input tensor (spatial coordinates).
            input_ch_views (int): Number of channels for view direction.
            output_ch (int): Number of output channels (density + RGB color).
        """
        super().__init__()

        # Define the first part of the MLP for the spatial coordinates
        layers = [nn.Linear(input_ch, W)]
        for _ in range(D - 1):
            layers += [nn.ReLU(inplace=True), nn.Linear(W, W)]

        # The final layer of the density network
        self.mlp_density = nn.Sequential(*layers, nn.Linear(W, 1))

        # Define the second part of the MLP for the color
        # This part includes view direction as additional input
        layers = [nn.ReLU(inplace=True), nn.Linear(W + input_ch_views, W // 2)]
        for _ in range(D - 2):
            layers += [nn.ReLU(inplace=True), nn.Linear(W // 2, W // 2)]

        # The final layer of the color network
        self.mlp_color = nn.Sequential(*layers, nn.Linear(W // 2, 3))

    def forward(self, x, view_dir):
        """
        Forward pass of NeRF.

        Args:
            x (torch.Tensor): Input tensor for spatial coordinates.
            view_dir (torch.Tensor): Input tensor for view direction.

        Returns:
            torch.Tensor: Output tensor of density and color.
        """
        # Process the input through the density network
        sigma = self.mlp_density(x)
        sigma = F.relu(sigma)  # Ensure the density is non-negative

        # Concatenate the output of the first MLP with view direction
        x = torch.cat([x, view_dir], -1)

        # Process through the color network
        rgb = self.mlp_color(x)
        rgb = torch.sigmoid(rgb)  # Ensure RGB values are between 0 and 1

        # Concatenate density and color
        output = torch.cat([rgb, sigma], -1)
        return output


