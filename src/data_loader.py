import json
import os
import numpy as np
import json
from PIL import Image
from sympy import im
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.utils import *

#old version
class SynDataset(Dataset):
    """
    Dataset class for synthetic data used in Novel View Synthesis with NeRF.
    
    Args:
        obj_name (str): Name of the object.
        root_dir (str): Root directory of the dataset.
        split (str, optional): Split of the dataset (train, val, test). Defaults to 'train'.
        transform (callable, optional): Optional transform to be applied to the image. Defaults to None.
    
    """
    def __init__(self, obj_name, root_dir, split='train', img_size=400, min_max=None, num_points=64):
        assert split in ['train', 'val', 'test'], "Invalid split, options are: train, val, test"
        data_folder = os.path.join(root_dir, "data", "nerf_synthetic", obj_name)
        json_file = os.path.join(data_folder, f"transforms_{split}.json")
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.root_dir = data_folder

        self.num_points = num_points

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size))]
            )
        self.camera_angle_x = self.data['camera_angle_x']
        self.camera_angle_x = torch.tensor(self.camera_angle_x)
        self.focal_length = 0.5 * img_size / torch.tan(0.5 * self.camera_angle_x)
        #store all data in tensors
        self.images = []
        self.transform_matrices = []

        for frame in self.data['frames']:
            img_path = frame['file_path'][2:] + '.png'
            img_path = os.path.join(self.root_dir, img_path,)
            # print(img_path)
            image = Image.open(img_path)
            image = transforms.ToTensor()(image)[:3] #shape: (C , img_size, img_size)
            self.images.append(image)
   

            # Extract camera parameters
            transform_matrix = torch.tensor(frame['transform_matrix'])
            self.transform_matrices.append(transform_matrix)

        self.images = torch.stack(self.images) #shape: (num_frames, 3, img_size, img_size)
        self.images = self.transform(self.images)

        self.transform_matrices = torch.stack(self.transform_matrices) #shape: (num_frames, 4, 4)
        self.rotations = self.transform_matrices[:, :3, :3] #shape: (num_frames, 3, 3)
        self.locations = self.transform_matrices[:, :3, 3].view(-1, 3) #shape: (num_frames, 3)
        
        self.min_max = None

        if split != 'train':
            self.min_max = min_max
        else:
            self.min_max = torch.zeros((2, 3))
            self.min_max[0] = torch.min(self.locations, dim=0)[0]
            self.min_max[1] = torch.max(self.locations, dim=0)[0]

        #normalize x, y, z to [-1, 1]
        self.locations = (self.locations - self.min_max[0]) / (self.min_max[1] - self.min_max[0])*2 - 1
        

    def __len__(self):
        """
        Returns the total number of frames in the dataset.
        
        Returns:
            int: Total number of frames.
        """
        return len(self.data['frames'])

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            dict: A dictionary containing the image, 5D scene representation(x, y, z, theta(azimuthal angle), phi(elevation angle)).
            x, y, z are normalized to [0, 1]

        """
        
        rays_o, rays_d = get_rays(self.images[idx], self.locations[idx], self.rotations[idx], self.focal_length)
        points, z_vals = sample_points(rays_o, rays_d, self.num_points)

        v_dir = dir_to_euler(rays_d)

        sample = {
            'img': self.images[idx], #shape: [C, H, W]
            # 'rays_o': rays_o, #shape: [H * W, 3]
            # 'rays_d': rays_d, #shape: [H * W, 3]
            'points': points, #shape: [H * W, N_samples, 3]
            'z_vals': z_vals, #shape: [H * W * N_samples, 1]
            'v_dir': v_dir, #shape: [H * W, 2]
        }

        return sample



#batch size unit = rays instead of frames
class SynDatasetRay(Dataset):
    def __init__(self, obj_name, root_dir, split='train', img_size=400, min_max=None, num_points=64):
        """
        Dataset class for synthetic data used in Novel View Synthesis with NeRF.
        
        Args:
            obj_name (str): Name of the object.
            root_dir (str): Root directory of the dataset.
            split (str, optional): Split of the dataset (train, val, test). Defaults to 'train'.
        
        """
        assert split in ['train', 'val', 'test'], "Invalid split, options are: train, val, test"
        data_folder = os.path.join(root_dir, "data", "nerf_synthetic", obj_name)
        json_file = os.path.join(data_folder, f"transforms_{split}.json")
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.root_dir = data_folder
        self.image_size = img_size
        self.num_points = num_points

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size))]
            )
        self.camera_angle_x = self.data['camera_angle_x']
        self.camera_angle_x = torch.tensor(self.camera_angle_x)
        self.focal_length = 0.5 * img_size / torch.tan(0.5 * self.camera_angle_x)
        #store all data in tensors
        self.images = []
        self.transform_matrices = []

        for frame in self.data['frames']:
            img_path = frame['file_path'][2:] + '.png'
            img_path = os.path.join(self.root_dir, img_path,)
            image = Image.open(img_path)
            image = transforms.ToTensor()(image)[:3] #shape: (C , img_size, img_size)
            self.images.append(image)
   

            # Extract camera parameters
            transform_matrix = torch.tensor(frame['transform_matrix'])
            self.transform_matrices.append(transform_matrix)

        self.images = torch.stack(self.images) 
        self.images = self.transform(self.images) #shape: (num_frames, 3, img_size, img_size)

        self.transform_matrices = torch.stack(self.transform_matrices) #shape: (num_frames, 4, 4)
        self.rotations = self.transform_matrices[:, :3, :3] #shape: (num_frames, 3, 3)
        self.locations = self.transform_matrices[:, :3, 3].view(-1, 3) #shape: (num_frames, 3)
        
        self.min_max = None

        if split != 'train':
            self.min_max = min_max
        else:
            self.min_max = torch.zeros((2, 3))
            self.min_max[0] = torch.min(self.locations, dim=0)[0]
            self.min_max[1] = torch.max(self.locations, dim=0)[0]

        #normalize x, y, z to [-1, 1]
        self.locations = (self.locations - self.min_max[0]) / (self.min_max[1] - self.min_max[0])*2 - 1

        self.current_frame = 0
        self.rays_o, self.rays_d = get_rays(self.images[self.current_frame], 
                                            self.locations[self.current_frame], 
                                            self.rotations[self.current_frame], 
                                            self.focal_length)
        self.points, self.z_vals = sample_points(self.rays_o, self.rays_d, self.num_points)
        self.v_dir = dir_to_euler(self.rays_d)
        self.current_img = self.images[self.current_frame].permute(1, 2, 0).view(-1, 3) #shape: [H * W, C]



    def __len__(self):
        """
        Returns the total number of frames in the dataset.
        
        Returns:
            int: Total number of frames.
        """
        return len(self.data['frames']) * self.image_size * self.image_size

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            Dict: A dict containing the ray origin, ray direction, sampled points, their corresponding depths (z values), and rgb values.

        """

        frame_idx = idx // (self.image_size * self.image_size)
        ray_idx = idx % (self.image_size * self.image_size)

        # Only compute rays for a new frame
        if frame_idx != self.current_frame:
            # print('idx: ', idx)
            # print("frame_idx: ", frame_idx)
            self.current_frame = frame_idx
            self.rays_o, self.rays_d = get_rays(self.images[self.current_frame], 
                                                self.locations[self.current_frame], 
                                                self.rotations[self.current_frame], 
                                                self.focal_length)
            self.points, self.z_vals = sample_points(self.rays_o, self.rays_d, self.num_points)
            self.v_dir = dir_to_euler(self.rays_d)
            self.current_img = self.images[self.current_frame].permute(1, 2, 0).view(-1, 3) # Reshape to [H * W, C]
            

        sample = {
            'rays_o': self.rays_o[ray_idx], #shape: [3]
            'rays_d': self.rays_d[ray_idx], #shape: [3]
            'points': self.points[ray_idx], #shape: [N_samples, 3]
            'z_vals': self.z_vals[ray_idx], #shape: [N_samples, 1]
            'v_dir': self.v_dir[ray_idx], #shape: [2]
            'rgb': self.current_img[ray_idx] #shape: [3]
        }

        return sample
