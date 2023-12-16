#Implementation of the NeRF model

from sympy import N
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

PI = math.pi

#old version, not used
def batched_position_encoding(points, v_dir, L_p=10, L_v=4):
    """
    Batched position encoding for the points and view directions.
    :param points: Sampled points. Shape: [B, H * W, N_samples, 3]
    :param v_dir: View directions. Shape: [B, H * W, 2]
    :param L_p: Number of frequency bands for point position encoding.
    :param L_v: Number of frequency bands for view direction encoding.
    :return: Encoded points and view directions.
    """
    B = points.shape[0]  # Batch size
    HW = points.shape[1]  # Number of pixels
    N_samples = points.shape[2]  # Number of samples

    # Frequencies for position encoding
    freqs_p = torch.tensor([(2**i)*PI for i in range(L_p)], dtype=torch.float32, device=points.device)
    freqs_p = freqs_p.view(1, 1, 1, 1, -1)  # Shape: [1, 1, 1, 1, L_p]

    # Position encoding for points
    points = points.view(B, HW, N_samples, 3, 1) * freqs_p  # Shape: [B, H * W, N_samples, 3, L_p]
    points_encoded = torch.cat([torch.sin(points), torch.cos(points)], dim=-1) 
    points_encoded = points_encoded.view(B, HW, N_samples, -1) # Shape: [B, H * W, N_samples, 3 * 2 * L_p]

    # Frequencies for view direction encoding
    freqs_v = torch.tensor([(2**i)*PI for i in range(L_v)], dtype=torch.float32, device=v_dir.device)
    freqs_v = freqs_v.view(1, 1, 1, -1)  # Shape: [1, 1, 1, L_v]

    # Position encoding for view directions
    v_dir = v_dir.view(B, HW, 2, 1) * freqs_v  # Shape: [B, H * W, 2, L_v]
    v_dir_encoded = torch.cat([torch.sin(v_dir), torch.cos(v_dir)], dim=-1)
    v_dir_encoded = v_dir_encoded.view(B, HW, -1) # Shape: [B, H * W, 2 * 2 * L_v]

    return points_encoded, v_dir_encoded



def position_encoding(points, v_dir, L_p=10, L_v=4):
    """
    Position encoding for the points and view directions.

    Args:
        points (torch.Tensor): Sampled points. Shape: [B, N_samples, 3]
        v_dir (torch.Tensor): View directions. Shape: [B, 3]
        L_p (int): Number of frequency bands for point position encoding. Default is 10.
        L_v (int): Number of frequency bands for view direction encoding. Default is 4.

    Returns:
        torch.Tensor: Encoded points. Shape: [B, N_samples, 3 * 2 * L_p]
        torch.Tensor: Encoded view directions. Shape: [B, 3 * 2 * L_v]
    """
    # Frequencies for position encoding
    B = points.shape[0]  # Batch size
    N_samples = points.shape[1]  # Number of samples

    freqs_p = torch.tensor([(2**i)*PI for i in range(L_p)], dtype=torch.float32, device=points.device)
    freqs_p = freqs_p.view(1, -1)  # Shape: [1, L_p]

    # Position encoding for points
    points = points.view(-1, 1) * freqs_p  # Shape: [B * N_samples * 3, L_p]
    # print('points', points.shape)
    points_encoded = torch.cat([torch.sin(points), torch.cos(points)], dim=-1).view(B, N_samples, -1) # Shape: [B, N_samples, 3 * 2 * L_p]
    # print('points_encoded', points_encoded.shape)

    # Frequencies for view direction encoding
    freqs_v = torch.tensor([(2**i)*PI for i in range(L_v)], dtype=torch.float32, device=v_dir.device)
    freqs_v = freqs_v.view(1, -1)  # Shape: [1, L_v]

    # Position encoding for view directions
    v_dir = v_dir.view(-1, 1) * freqs_v # Shape: [B * 2, L_v]
    v_dir_encoded = torch.cat([torch.sin(v_dir), torch.cos(v_dir)], dim=-1).view(B, -1)  # Shape: [B, 3 * 2 * L_v]

    return points_encoded, v_dir_encoded



class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch_pos=3, input_ch_dir=3, L_p=10, L_v=4, skips=[4]):
        """
        NeRF Network with integrated position encoding.

        Args:
            D (int): Number of layers (default: 8).
            W (int): Number of neurons in hidden layers (default: 256).
            input_ch_pos (int): Number of input channels for position (default: 3).
            input_ch_dir (int): Number of input channels for view direction (default: 3).
            L_p (int): Number of frequency bands for position encoding.
            L_v (int): Number of frequency bands for view direction encoding.
            skips (list of int): Layers to add skip connections (default: [4]).
        """
        super().__init__()

        self.D = D
        self.W = W
        self.input_ch_pos = input_ch_pos
        self.input_ch_dir = input_ch_dir
        self.L_p = L_p
        self.L_v = L_v
        self.skips = skips

        # Adjusted input dimensions based on position encoding
        encoded_pos_dims = input_ch_pos * 2 * L_p
        encoded_dir_dims = input_ch_dir * 2 * L_v

        # Fully connected layers for position encoding
        self.fc_pos = nn.ModuleList([nn.Linear(encoded_pos_dims, W)] + [nn.Linear(W, W) if i not in skips else nn.Linear(W + encoded_pos_dims, W) for i in range(D-1)])

        # Output layer for density (sigma) and feature vector
        self.fc_sigma = nn.Linear(W, 1)
        self.fc_feature = nn.Linear(W, W)

        # View direction layers
        self.fc_dir = nn.Linear(encoded_dir_dims + W, W//2)
        self.fc_rgb = nn.Linear(W//2, 3)

    def forward(self, x, d):
        """
        Forward pass of the model with position encoding.

        Args:
            x (torch.Tensor): Input positions. Shape: [B, N_samples, 3]
            d (torch.Tensor): Input view directions. Shape: [B, 2]

        Returns:
            torch.Tensor: RGB color and density.
        """
        # Position encoding for points and directions
        x_encoded, d_encoded = position_encoding(x, d, self.L_p, self.L_v)

        N_samples = x_encoded.shape[1]  # Number of samples

        # Flatten the encoded inputs for processing in fully connected layers
        x_encoded = x_encoded.view(-1, x_encoded.shape[-1]) # [B * N_samples, 3 * 2 * L_p]
        d_encoded = d_encoded.view(-1, d_encoded.shape[-1]).unsqueeze(1).repeat(1, N_samples, 1).view(-1, d_encoded.shape[-1]) # [B * N_samples, 2 * 2 * L_v]
        

        input_x_encoded = x_encoded # [B * N_samples, 3 * 2 * L_p]

        # Position encoding layers
        for i, layer in enumerate(self.fc_pos):
            x_encoded = F.relu(layer(x_encoded)) # [B * N_samples, 3 * 2 * L_p]
            if i in self.skips:
                x_encoded = torch.cat([x_encoded, input_x_encoded], -1) # [B * N_samples, W + 3 * 2 * L_p]
              

        # Output density and feature vector
        sigma = F.relu(self.fc_sigma(x_encoded)) # [B * N_samples, 1]
        feature = self.fc_feature(x_encoded) # [B * N_samples, W]

        # View direction layers
        d_encoded = torch.cat([feature, d_encoded], -1) # [B * N_samples, W + encoded_dir_dims]
        d_encoded = F.relu(self.fc_dir(d_encoded)) # [B * N_samples, W//2]

        # RGB color
        rgb = torch.sigmoid(self.fc_rgb(d_encoded))

        return rgb.view(-1, N_samples, 3), sigma.view(-1, N_samples)