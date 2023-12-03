#Implementation of the NeRF model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

PI = math.pi

def position_encoding(points, v_dir, L_p=10, L_v=4):
    """
    Position encoding for the points and view directions.
    :param points: Sampled points. Shape: [H * W, N_samples, 3]
    :param v_dir: View directions. Shape: [H * W, 2]
    :param L_p: Number of frequency bands for point position encoding.
    :param L_v: Number of frequency bands for view direction encoding.
    :return: Encoded points and view directions.
    """
    # Frequencies for position encoding

    # print('points.shape', points.shape)
    # print('v_dir.shape', v_dir.shape)

    freqs_p = torch.tensor([(2**i)*PI for i in range(L_p)], dtype=torch.float32, device=points.device)
    freqs_p = freqs_p.view(1, -1)  # Shape: [1, L_p]

    # Position encoding for points
    points = points.view(-1, 1) * freqs_p  # Shape: [H * W * N_samples * 3, L_p]
    points_encoded = torch.cat([torch.sin(points), torch.cos(points)], dim=-1) # Shape: [H * W * N_samples, 2 * L_p]


    # Frequencies for view direction encoding
    freqs_v = torch.tensor([(2**i)*PI for i in range(L_v)], dtype=torch.float32, device=v_dir.device)
    freqs_v = freqs_v.view(1, -1)  # Shape: [1, L_v]

    # Position encoding for view directions
    v_dir = v_dir.view(-1, 1) * freqs_v # Shape: [H * W * 2, L_v]
    v_dir_encoded = torch.cat([torch.sin(v_dir), torch.cos(v_dir)], dim=-1)  # Shape: [H * W, 2 * L_v]

    return points_encoded, v_dir_encoded


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


    





   

