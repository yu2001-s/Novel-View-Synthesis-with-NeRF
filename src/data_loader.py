import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SynDataset(Dataset):
    """
    Dataset class for synthetic data used in Novel View Synthesis with NeRF.
    
    Args:
        obj_name (str): Name of the object.
        root_dir (str): Root directory of the dataset.
        split (str, optional): Split of the dataset (train, val, test). Defaults to 'train'.
        transform (callable, optional): Optional transform to be applied to the image. Defaults to None.
    
    """
    def __init__(self, obj_name, root_dir, split='train', transform=None, min_max=None):
        assert split in ['train', 'val', 'test'], "Invalid split, options are: train, val, test"
        data_folder = os.path.join(root_dir, "data", "nerf_synthetic", obj_name)
        json_file = os.path.join(data_folder, f"transforms_{split}.json")
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.root_dir = data_folder
        self.transform = transform
        
        #store all data in tensors
        self.images = []
        self.scene_5d = []
        for frame in self.data['frames']:
            img_path = frame['file_path'][2:] + '.png'
            img_path = os.path.join(self.root_dir, img_path,)
            image = Image.open(img_path)
            image = transforms.ToTensor()(image)
            self.images.append(image)

            # Extract camera parameters
            transform_matrix = torch.tensor(frame['transform_matrix'])
            position = transform_matrix[:3, 3]
            z_axis = -transform_matrix[:3, 2]  # Negative z-axis is the viewing direction
            theta = torch.arctan2(z_axis[1], z_axis[0])
            phi = torch.arccos(z_axis[2])
            scene_5d = torch.cat([position, theta.view(1), phi.view(1)])
            self.scene_5d.append(scene_5d)

        self.images = torch.stack(self.images)
        self.images = self.transform(self.images)
        self.scene_5d = torch.stack(self.scene_5d)
        
        self.min_max = None

        if split != 'train':
            self.min_max = min_max
        else:
            self.min_max = torch.zeros((2, 3))
            self.min_max[0] = torch.min(self.scene_5d, dim=0)[0][:3]
            self.min_max[1] = torch.max(self.scene_5d, dim=0)[0][:3]

        #normalize x, y, z to [-1, 1]
        self.scene_5d[:, :3] = (self.scene_5d[:, :3] - self.min_max[0]) / (self.min_max[1] - self.min_max[0])*2 - 1

    
    



        

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
        

        sample = {
            'image': self.images[idx],
            'scene_5d': self.scene_5d[idx]
        }

        return sample
