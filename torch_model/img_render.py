import copy
import os
import sys
sys.path.append('.')

import cv2
import numpy as np
import open3d as o3d
import torch

from camera import Camera
from sampling_branch import intersect_ray_aabb 
import model


model_name = "lego"
path_to_weigths = f"/home/diego/data/nerf/ckpt_syn/256_to_512_fasttv/{model_name}/ckpt.npz"
img_size = 800
batch_size = 1024*4
nb_samples = 512
nb_sh_channels = 3
size_model = 256
device = "cuda"

data = np.load(path_to_weigths, allow_pickle=True)

# Access data arrays using keys
npy_radius = data['radius']
npy_center = data['center']
npy_links = data['links']
npy_links = npy_links[::2, ::2, ::2] # reduce resolution to half - use only if size_model = 256
npy_density_data = data['density_data']
npy_sh_data = data['sh_data']
npy_basis_type = data['basis_type']

mask = npy_links >= 0
npy_links = npy_links[mask]

density_matrix = np.zeros((size_model, size_model, size_model, 1), dtype=np.float32)
density_matrix[mask] = npy_density_data[npy_links]
density_matrix = np.reshape(density_matrix, (size_model, size_model, size_model))
density_matrix = torch.from_numpy(density_matrix)

sh_matrix = np.zeros((size_model, size_model, size_model, 27), dtype=np.float16)
sh_matrix[mask] = npy_sh_data[npy_links]
sh_matrix = np.reshape(sh_matrix, (size_model, size_model, size_model, 27))
sh_matrix = torch.from_numpy(sh_matrix)

rf = model.RadianceField(idim=size_model, grid=sh_matrix, opacity=density_matrix, 
                         nb_sh_channels=nb_sh_channels, nb_samples=nb_samples,
                         delta_voxel=torch.tensor([1, 1, 1], dtype=torch.float),
                         device=device)

# load the scene now: for a camera, get the rays
# from ./train/r_13:
trans_mat = np.array(
        [
                [
                    0.842908501625061,
                    -0.09502744674682617,
                    0.5295989513397217,
                    2.1348819732666016
                ],
                [
                    0.5380570292472839,
                    0.14886793494224548,
                    -0.8296582698822021,
                    -3.3444597721099854
                ],
                [
                    7.450582373280668e-09,
                    0.9842804074287415,
                    0.17661221325397491,
                    0.7119466662406921
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
            )
camera = Camera(0.6911112070083618, trans_mat, int(img_size/2), int(img_size/2), size_model)
rays_dirs, rays_origins = camera.get_camera_ray_dirs_and_origins()
rays_origins = rays_origins.astype(np.float32)
rays_dirs = rays_dirs.astype(np.float32)

box_min = np.zeros(rays_origins.shape)
box_max = np.ones(rays_origins.shape)*(size_model - 2)
ray_inv_dirs = 1. / rays_dirs
tmin, tmax = intersect_ray_aabb(rays_origins, ray_inv_dirs, box_min, box_max)
mask = tmin < tmax
valid_rays_origins = torch.from_numpy(rays_origins[mask])
valid_rays_dirs = torch.from_numpy(rays_dirs[mask])
valid_tmin = torch.from_numpy(tmin[mask])
valid_tmax = torch.from_numpy(tmax[mask])

print("shape of rays to render:", valid_rays_origins.shape)
rendered_rays = rf.render_rays(valid_rays_origins, valid_rays_dirs, valid_tmin, valid_tmax, batch_size).numpy()
complete_colors = np.zeros((rays_origins.shape[0], 3))
complete_colors[mask] = rendered_rays
max_val = np.max(complete_colors)
min_val = np.min(complete_colors)
print("max px val:", max_val)
print("min px val:", min_val)
complete_colors[complete_colors > 1] = 1
complete_colors[complete_colors < 0] = 0

img = np.reshape(complete_colors, (img_size, img_size, nb_sh_channels))
img = (img * 255).astype(np.uint8)
if nb_sh_channels == 2:
    img = np.concatenate((img, np.zeros((img_size, img_size, 1)) + 0.5), axis=2)
elif nb_sh_channels == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(f"render_ex_ch{nb_sh_channels}_{img_size}x{img_size}_s{nb_samples}.png", img)
