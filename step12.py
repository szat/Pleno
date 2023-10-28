import copy
import os
import sys
sys.path.append('.')
import cv2
import numpy as np
from jax_helpers import *
from functools import partial
from jax import value_and_grad

model_name = "lego"
# path_to_weigths = f"/home/diego/data/nerf/ckpt_syn/256_to_512_fasttv/{model_name}/ckpt.npz"
path_to_weigths = f"/home/adrian/Documents/Nerf/256_to_512_fasttv/{model_name}/ckpt.npz"
img_size = 800
batch_size = 1024*4
nb_samples = 512
nb_sh_channels = 3
size_model = 128
device = "cuda"

data = np.load(path_to_weigths, allow_pickle=True)

# Access data arrays using keys
npy_radius = data['radius']
npy_center = data['center']
npy_links = data['links']
npy_density_data = data['density_data']
npy_sh_data = data['sh_data']
npy_basis_type = data['basis_type']

tmp = np.zeros([1, 28])
npy_data = np.hstack([npy_density_data, npy_sh_data])
npy_data = np.vstack([tmp, npy_data])

npy_links[npy_links >= 0] += 1
assert (npy_links == 0).sum() == 0
npy_links[npy_links < 0] = 0
assert (npy_links == 0).sum() + (npy_links > 0).sum() == 512**3
assert (npy_links > 0).sum() == len(npy_data) - 1 #since we did a 0 padding, collapse everything to 0

links2 = npy_links[::4, ::4, ::4]
npy_data2 = npy_data[np.unique(links2)]
assert (links2 > 0).sum() == len(npy_data2) - 1 #again cause of 0 padding
links2[links2 > 0] = np.arange(1, len(npy_data2))
assert len(np.unique(links2)) == len(npy_data2)

npy_data = npy_data2
npy_links = links2

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

mask = tmin < tmax
x = np.concatenate((rays_origins, rays_dirs, tmin[:, None], tmax[:, None], sh_mine), axis=1)
x = x[mask]

colors = np.zeros([800*800, 3])
max_dt = np.max(tmax - tmin)
nb = 100
step_size = 0.5
delta_scale = 1/128

tics = np.arange(0.05, 0.95, step=1/nb_samples)
tics = jnp.array(tics)

def render(x, npy_links_in, npy_data_in):
    ori = x[:3]
    dir = x[3:6]
    tmin = x[6:7]
    tmax = x[7:8]
    my_sh_in = x[8:]

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

render_jit = jit(vmap(render, in_axes=(0, None, None)))

batch_size = 4000
batch_nb = jnp.ceil(len(x) / batch_size)
tmp_rgb = []
for i in range(int(batch_nb - 1)):
    res = render_jit(x[i * batch_size: (i + 1) * batch_size], npy_links, npy_data)
    res.block_until_ready()
    tmp_rgb.append(res)
    print(i)

last_dab = len(x) - (batch_nb - 1) * batch_size
res = render_jit(x[int((batch_nb - 1) * batch_size):], npy_links, npy_data)
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



# training part
def mse(x, truth, links, params):
    pred = jnp.squeeze(render_jit(x, links, params))
    out = jnp.mean((pred - truth) ** 2)
    return out

i = 0
batch_size = 2000
res = render_jit(x[i * batch_size: (i + 1) * batch_size], npy_links, npy_data)
tmp = mse(x[i * batch_size: batch_size], res, npy_links, npy_data)

grad_mse = value_and_grad(mse, argnums=3)
grad_mse_batch = jit(grad_mse)

v, g = grad_mse_batch(x[i * batch_size: batch_size], res, npy_links, npy_data)

@jax.jit
def train_step(x, truth, links, params, learning_rate):
    loss, gradient = grad_mse_batch(x, truth, links, params)
    params = params - learning_rate * gradient
    return loss, params

mean = 3
std_dev = 2
# Generating Gaussian noise of shape (5463817, 28)
gaussian_noise = np.random.normal(mean, std_dev, npy_data.shape)
npy_data_start = npy_data + gaussian_noise

# i = 0
# batch_size = 2000
# res2 = render_jit(x[i * batch_size: (i + 1) * batch_size], npy_links, npy_data_start)
# tmp = mse(x[i * batch_size: batch_size], res, npy_links, npy_data_start)

learning_rate = 0.01
loss_hist = []
params = npy_data_start
for i in range(100):
    loss, params = train_step(x[:batch_size], res, npy_links, params, learning_rate)
    loss_hist.append(loss)
    print(i)

import matplotlib.pyplot as plt
plt.plot(loss_hist)
plt.show()





