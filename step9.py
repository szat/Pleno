import copy
import os
import sys
sys.path.append('.')
import cv2
import numpy as np
from jax_helpers import *

model_name = "lego"
# path_to_weigths = f"/home/diego/data/nerf/ckpt_syn/256_to_512_fasttv/{model_name}/ckpt.npz"
path_to_weigths = f"/home/adrian/Documents/Nerf/256_to_512_fasttv/{model_name}/ckpt.npz"
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
npy_data = np.hstack([npy_density_data, npy_sh_data])
mask = npy_links >= 0
npy_links_m = npy_links[mask]

origin = np.array([size_model + 20., size_model + 20., size_model + 20.])
orientation = np.array([-1., -1., -1.])
orientation = orientation / np.linalg.norm(orientation)
camera = Camera(origin=origin, orientation=orientation, dist_plane=2, length_x=1, length_y=1,
                pixels_x=img_size, pixels_y=img_size)
rays_cam = get_camera_rays(camera)
rays_origins = np.tile(origin, (img_size*img_size, 1)).astype(np.float32)
rays_dirs = rays_cam.reshape((img_size*img_size, 3)).astype(np.float32)
box_min = np.zeros(rays_origins.shape)
box_max = np.ones(rays_origins.shape)*(size_model - 2)
ray_inv_dirs = 1. / rays_dirs
tmin, tmax = intersect_ray_aabb(rays_origins, ray_inv_dirs, box_min, box_max)
sh_mine = eval_sh_bases_mine(rays_dirs)

mask = np.ones([800, 800])
mask = mask == 1
mask = mask.flatten()

mask = tmin < tmax
valid_rays_origins = rays_origins[mask]
valid_rays_dirs = rays_dirs[mask]
valid_tmin = tmin[mask]
valid_tmax = tmax[mask]
valid_sh = sh_mine[mask]

colors = np.zeros([800*800, 3])
max_dt = np.max(tmax - tmin)
nb = 100
step_size = 0.5
delta_scale = 1/256

tics = np.arange(0.05, 0.95, step=1/nb_samples)
tics = jnp.array(tics)

def conv_to_vmap(ori, dir, tmin, tmax, my_sh_in, npy_links_in, npy_data_in):
    samples = (tmax - tmin) * tics + tmin
    tmp = jnp.matmul(samples[:, None], dir[None, :], precision='highest')
    tmp = jnp.add(tmp, ori[None, :])
    sample_points_in = tmp
    interp = jit_trilinear_interp(sample_points_in, npy_links_in, npy_data_in)
    # out = interp
    interp = np.squeeze(interp)
    interp_sh_coeffs = interp[:, 1:][None, :, :]
    interp_opacities = interp[:, :1][None, :, :]
    interp_opacities = jnp.clip(interp_opacities, a_min=0.0, a_max=100000)
    deltas = samples[1:] - samples[:-1]

    interp_sh_coeffs = interp_sh_coeffs.reshape(1, samples.shape[0], 3, 9)
    interp_opacities = interp_opacities.reshape(1, samples.shape[0], 1)
    interp_harmonics = interp_sh_coeffs * my_sh_in[None, None, None, :]

    interp_opacities = jnp.squeeze(interp_opacities)

    deltas_times_sigmas = - deltas * interp_opacities[:-1]

    cum_weighted_deltas = jnp.cumsum(deltas_times_sigmas)
    cum_weighted_deltas = jnp.concatenate([jnp.zeros(1), cum_weighted_deltas[:-1]])

    samples_colors = jnp.clip(jnp.sum(interp_harmonics, axis=3) + 0.5, a_min=0.0, a_max=100000)
    samples_colors = jnp.squeeze(samples_colors)
    deltas_times_sigmas = jnp.squeeze(deltas_times_sigmas)
    tmp1 = jnp.exp(cum_weighted_deltas)
    tmp2 = 1 - jnp.exp(deltas_times_sigmas)
    rays_color = jnp.sum(tmp1[:, None] * tmp2[:, None] * samples_colors[:-1], axis=0)
    out = rays_color
    return out

jit_again = jit(vmap(conv_to_vmap, in_axes=(0, 0, 0, 0, 0, None, None)))

batch_size = 5000
batch_nb = jnp.ceil(len(valid_rays_dirs) / batch_size)

tmp_rgb = []
for i in range(int(batch_nb - 1)):
    res = jit_again(valid_rays_origins[i * batch_size: (i + 1) * batch_size],
                   valid_rays_dirs[i * batch_size: (i + 1) * batch_size],
                   valid_tmin[i * batch_size: (i + 1) * batch_size],
                   valid_tmax[i * batch_size: (i + 1) * batch_size],
                   valid_sh[i * batch_size: (i + 1) * batch_size],
                   npy_links, npy_data)
    res.block_until_ready()
    tmp_rgb.append(res)

last_dab = len(valid_rays_dirs) - (batch_nb - 1) * batch_size
res = jit_again(valid_rays_origins[int((batch_nb - 1) * batch_size):],
                valid_rays_dirs[int((batch_nb - 1) * batch_size):],
                valid_tmin[int((batch_nb - 1) * batch_size):],
                valid_tmax[int((batch_nb - 1) * batch_size):],
                valid_sh[int((batch_nb - 1) * batch_size):],
                npy_links, npy_data)
tmp_rgb.append(res)
colors = np.concatenate(tmp_rgb)

complete_colors = np.zeros((rays_origins.shape[0], 3))
complete_colors[mask] = colors

complete_colors[complete_colors > 1] = 1
complete_colors[complete_colors < 0] = 0

img = complete_colors.reshape([800,800,3])
import cv2
img = (img * 255).astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("hello")

