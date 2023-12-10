import torch

def get_rays(image, camera_pos, camera_rot, focal_length):
    """
    Calculate the origin and direction of the ray for each pixel in the image.

    Args:
        image: The input image. [C, H, W]
        camera_pos: Camera position as a PyTorch tensor. [3]
        camera_rot: Camera rotation matrix as a PyTorch tensor. [3, 3]
        focal_length: Focal length of the camera.

    Returns:
        rays_o (ray origins), rays_d (ray directions) [H * W, 3]
    """
    C, H, W = image.shape  # image is a PyTorch tensor of shape [C, H, W]

    # Create a meshgrid for pixel coordinates
    y, x = torch.meshgrid(torch.arange(0, H, dtype=torch.float32),
                        torch.arange(0, W, dtype=torch.float32),
                        indexing='ij')

    # Normalize pixel coordinates
    x = (x - W * 0.5) / focal_length
    y = (y - H * 0.5) / focal_length

    # Create direction vectors in camera coordinates
    dirs = torch.stack([x, -y, -torch.ones_like(x)], dim=-1)  # Flipping y due to image coordinates, shape [H, W, 3]

    # Reshape and transform to world coordinates
    dirs = dirs.reshape(-1, 3)
    rays_d = torch.matmul(dirs, camera_rot.transpose(0, 1)) # shape [H * W, 3]

    # Normalize the ray directions
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)

    # The origin of all rays is the camera position
    rays_o = camera_pos.expand(H * W, -1) # shape [H * W, 3]

    return rays_o, rays_d



def sample_points(rays_o, rays_d, N_samples):
    """
    Sample points along the rays within a normalized bounding volume.

    Args:
        rays_o: Ray origins. Shape: [H * W, 3]
        rays_d: Ray directions. Shape: [H * W, 3]
        N_samples: Number of samples to take along each ray.

    Returns:
        Sampled points and their corresponding depths (z values)
    """
    # find length of each ray
    # find -1 or 1 for each ray by comparing the sign of the ray direction
    # if the ray direction is negative, we want to go to -1, else 1

    # find the sign of the ray direction
    signs = torch.sign(rays_d)  # shape: [H * W, 3]
    bounds = torch.where(signs < 0, -1, 1)  # shape: [H * W, 3]

    # find the length
    lengths = (bounds - rays_o) / (rays_d)
    lengths = torch.where(torch.abs(rays_d) > 1e-6, lengths, float('inf'))  # handle division by zero

    lengths = torch.min(lengths, dim=1)[0]  # shape: [H * W]



    # find the z values and apply stratified sampling
    bin_len = 1.0 / N_samples
    z_vals = torch.linspace(0 + bin_len / 2, 1 - bin_len / 2, N_samples)  # [N_samples]
    # print('z_vals', z_vals)
    # print('bin_len', bin_len)
    # print()
    z_vals = torch.ger(lengths, z_vals)  # shape: [H * W, N_samples]
    noise = (torch.rand_like(z_vals) - 0.5) * bin_len * 2
    z_vals += noise


    rays_o = rays_o.unsqueeze(1).expand(-1, N_samples, 3)  # shape [H * W, N_samples, 3]
    rays_d = rays_d.unsqueeze(1).expand(-1, N_samples, 3)  # shape [H * W, N_samples, 3]

    z_vals = z_vals.unsqueeze(-1)  # shape [H * W, N_samples, 1]

    points = rays_o + rays_d * z_vals # shape [H * W, N_samples, 3]

    return points, z_vals

def dir_to_euler(ray_d):
    """
    Convert a direction vector to euler angles.

    Args:
        ray_d: Direction vector. Shape: [H * W, 3]
    
    Returns:
        Euler angles (θ, φ) in radians. Shape: [H * W, 2]
    """
    x = ray_d[:, 0]
    y = ray_d[:, 1]
    z = ray_d[:, 2]

    # Azimuthal angle (θ)
    theta = torch.atan2(y, x)

    # Polar angle (φ)
    phi = torch.atan2(torch.sqrt(x**2 + y**2), z)

    return torch.stack([theta, phi], dim=1)  # shape: [H * W, 2]
    

def volume_rendering(z_vals, rgb, sigma, white_bkgd=False):
    """
    Volume rendering function for NeRF.

    Args:
        z_vals (torch.Tensor): Depths of the sampled points. Shape: [num_rays, num_samples]
        rgb (torch.Tensor): Predicted RGB colors at sampled points. Shape: [num_rays, num_samples, 3]
        sigma (torch.Tensor): Predicted densities (sigma) at sampled points. Shape: [num_rays, num_samples]
        white_bkgd (bool): If True, render with a white background; otherwise, render with a black background.

    Returns:
        torch.Tensor: Rendered colors for each ray. Shape: [num_rays, 3]
    """
    # print('z_vals', z_vals.shape)
    # print('rgb', rgb.shape)
    # print('sigma', sigma.shape)
    # print()


    device = z_vals.device
    # Calculate distances between adjacent samples along the ray
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    # The last delta is infinity
    delta_inf = torch.Tensor([1e10]).expand_as(deltas[:, :1]).to(device)  # Very large value
    deltas = torch.cat([deltas, delta_inf], -1)

    # Calculate the alpha value for each sample point
    alpha = 1.0 - torch.exp(-sigma * deltas)
    T = torch.cumprod(1.0 - alpha + 1e-10, -1)[:, :-1]
    T = torch.cat([torch.ones_like(T[:, :1]), T], -1)  # [num_rays, num_samples]

    # Calculate weights for RGB colors
    weights = alpha * T

    # Calculate the final color of each ray
    rendered_rgb = (weights[:, :, None] * rgb).sum(dim=1)

    # If white background is used, add the background color
    if white_bkgd:
        rendered_rgb += (1.0 - weights.sum(dim=1))[:, None]

    return rendered_rgb
    