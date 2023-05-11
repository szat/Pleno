import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
from jax import device_put
import jax
import time
import contextlib
import timeit

from time import time

import numpy as np
from trilinear_interpolation import *
from sampling_branch import intersect_ray_aabb
from spherical_harmonics import eval_sh_bases_mine

path = '/home/adrian/Code/svox2/opt/ckpt/exp2/ckpt.npz'
img_size = 800
# batch_size = 4*1024
# nb_samples = 512
# nb_sh_channels = 3
data = np.load(path, allow_pickle=True)
npy_radius = data['radius']
npy_center = data['center']
npy_links = data['links']
npy_density_data = data['density_data']
npy_sh_data = data['sh_data']
npy_basis_type = data['basis_type']

# hack
npy_density_data[0] = 0
npy_sh_data[0] = 0
npy_links[npy_links < 0] = 0
npy_data = np.hstack([npy_density_data, npy_sh_data])

ori = np.load("/home/adrian/Documents/temp/svox_ori.npy")
dir = np.load("/home/adrian/Documents/temp/svox_dir.npy")

# transform to world coords
step_size = 0.5
delta_scale = 1/256
gsz = 256
offset = 0.5 * gsz - 0.5  # 0.5 * 256 - 0.5
scaling = 0.5 * gsz  # 0.5 * 256
ori = offset + ori * scaling

# get the tic samples
spacing = 0.5
box_top = np.ones(3)*256
box_bottom = np.zeros(3)
dir = dir / np.expand_dims(np.linalg.norm(dir, axis=1), 1)
inv = 1/dir
sh = eval_sh_bases_mine(dir)
tmin, tmax = intersect_ray_aabb(ori, inv, box_bottom-0.5, box_top-0.5) # to compute the tmin and tmax, they do a offset of -0.5 first, so add -0.5 to box_min and box_max

mask = tmin < tmax
ori = ori[mask]
dir = dir[mask]
sh = sh[mask]
tmin = tmin[mask]
tmax = tmax[mask]

colors = np.zeros([800*800, 3])
max_dt = np.max(tmax - tmin)

# for i in range(len(tmin)):
#     tics.append(np.linspace(tmin[i], tmax[i], nb))
#     tics.append(np.arange(tmin[i], tmax[i], spacing))

# t0 = time()
# for i in range(800*800):
#     # i = 65000
#     x = max_dt + tmin[i] - tmax[i]
#     tics = np.arange(tmin[i], tmax[i] + x, spacing)
#     samples = ori[i, None] + tics[:, None] * dir[i, None]
#     samples = np.clip(samples, 0, 254)
#     # interp = trilinear_interpolation_shuffle_zero(samples, npy_links, npy_data)
# t1 = time()
# print(t1 - t0)


nb = int(np.ceil(max_dt/spacing))
tmp_tics = np.linspace(0, 1, nb)
samples_list = []
weights_list = []
coeff_list = []
l000_list = []
for i in range(1000):
    # i = 65000
    # x = max_dt + tmin[i] - tmax[i]
    # tics = np.arange(tmin[i], tmax[i] + x, spacing)
    tics = tmin[i] + tmp_tics * (tmax[i] - tmin[i])

    print(len(tics))
    samples = ori[i, None] + tics[:, None] * dir[i, None]
    samples = np.clip(samples, 0, 254)
    samples_list.append(samples)
    # interp = trilinear_interpolation_shuffle_zero(samples, npy_links, npy_data)

    ###
    vecs = samples
    values_compressed = npy_data
    origin = np.zeros(3)
    delta_voxel = np.ones(3)
    links = npy_links

    if vecs.ndim == 1:
        vecs = vecs[None, :]

    dims = values_compressed.shape[1]
    nb_samples = vecs.shape[0]
    res = np.zeros([nb_samples, dims])

    xyz = vecs - origin
    xyz = xyz / delta_voxel
    xyz_floor = np.floor(xyz)
    diff = xyz - xyz_floor
    xd, yd, zd = diff[:, 0], diff[:, 1], diff[:, 2]

    xyz_floor = xyz_floor.astype(int)
    x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]

    l000 = links[x0, y0, z0]

    l100 = links[x0+1, y0, z0]
    l010 = links[x0, y0+1, z0]
    l001 = links[x0, y0, z0+1]
    l110 = links[x0+1, y0+1, z0]
    l011 = links[x0, y0+1, z0+1]
    l101 = links[x0+1, y0, z0+1]
    l111 = links[x0+1, y0+1, z0+1]

    v000 = values_compressed[l000]
    v100 = values_compressed[l100]
    v010 = values_compressed[l010]
    v001 = values_compressed[l001]
    v110 = values_compressed[l110]
    v011 = values_compressed[l011]
    v101 = values_compressed[l101]
    v111 = values_compressed[l111]

    tmpX = 1 - xd
    tmpY = 1 - yd
    tmpZ = 1 - zd
    a000 = tmpX * tmpY
    a100 = xd * tmpY
    a010 = tmpX * yd
    a110 = xd * yd
    weights = np.array([a000, a010, a100, a110])
    weights_list.append(weights)
    coeff = np.array([v000, v001, v010, v011, v100, v101, v110, v111])
    coeff_list.append(coeff)
    weights = weights[:, :, None]
    if tmpZ.ndim == 1 and zd.ndim == 1:
        tmpZ = tmpZ[:, None]
        zd = zd[:, None]

    out = np.sum(weights * coeff[[0, 2, 4, 6]], axis=0) * tmpZ + np.sum(weights * coeff[[1, 3, 5, 7]], axis=0) * zd
    interp = out
    ###

    sigma = interp[:, :1]
    rgb = interp[:, 1:]

    sigma = np.clip(sigma, a_min=0.0, a_max=100000)
    rgb = rgb.reshape(-1, 3, 9)

    sh_ray = sh[i][None, None, :]
    rgb = rgb * sh_ray
    rgb = np.sum(rgb, axis=2)
    rgb = rgb + 0.5 #correction 1
    rgb = np.clip(rgb, a_min=0.0, a_max=100000)
    tmp = step_size * sigma * delta_scale
    # tmp = np.clip(tmp, a_min=0.0, a_max=100000)
    var = 1 - np.exp(-tmp)
    Ti = np.exp(np.cumsum(-tmp))
    Ti = Ti[:, None]
    coefs = Ti * var
    rgb = coefs * rgb
    rgb = np.sum(rgb, axis=0)
    colors[i] = rgb
    print(i)

img = colors.reshape([800,800,3])
import cv2
# if nb_sh_channels == 2:
#     img = np.concatenate((img, np.zeros((img_size, img_size, 1)) + 0.5), axis=2)
img = (img * 255).astype(np.uint8)
# if nb_sh_channels == 3:
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


nb = int(np.ceil(max_dt/spacing))
tmin = tmin[:, None]
tmax = tmax[:, None]
ori = jnp.array(ori)
dir = jnp.array(dir)
tmin = jnp.array(tmin)
tmax = jnp.array(tmax)
tics = jnp.linspace(0, 1, nb)
tics = tics[:, None]
origin = jnp.zeros(3)
delta_voxel = jnp.ones(3)
sh = jnp.array(sh)
links = jnp.array(npy_links).astype(int)
values_compressed = jnp.array(npy_data).astype(jnp.float16)
tmp_tics = jnp.linspace(0, 1, nb)

def step1(ori, dir, tmin, tmax, sh):
    tics = tmin + tmp_tics * (tmax - tmin)
    ori = ori[None, :]
    dir = dir[None, :]
    tics = tics[:, None]
    print(tics.shape)
    print(ori.shape)
    print(dir.shape)
    samples = ori + dir * tics
    samples = jnp.clip(samples, 0, 254)
    # return samples
    vecs = samples
    xyz = vecs - origin
    xyz = xyz / delta_voxel
    xyz_floor = jnp.floor(xyz)
    diff = xyz - xyz_floor
    xd, yd, zd = diff[:, 0], diff[:, 1], diff[:, 2]
    xyz_floor = xyz_floor.astype(int)
    x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]
    l000 = links[x0, y0, z0]
    l100 = links[x0+1, y0, z0]
    l010 = links[x0, y0+1, z0]
    l001 = links[x0, y0, z0+1]
    l110 = links[x0+1, y0+1, z0]
    l011 = links[x0, y0+1, z0+1]
    l101 = links[x0+1, y0, z0+1]
    l111 = links[x0+1, y0+1, z0+1]
    v000 = values_compressed[l000]
    v100 = values_compressed[l100]
    v010 = values_compressed[l010]
    v001 = values_compressed[l001]
    v110 = values_compressed[l110]
    v011 = values_compressed[l011]
    v101 = values_compressed[l101]
    v111 = values_compressed[l111]
    tmpX = 1 - xd
    tmpY = 1 - yd
    tmpZ = 1 - zd
    a000 = tmpX * tmpY
    a100 = xd * tmpY
    a010 = tmpX * yd
    a110 = xd * yd

    weights = jnp.array([a000, a010, a100, a110])
    # return weights
    coeff = jnp.array([v000, v001, v010, v011, v100, v101, v110, v111])
    weights = jnp.expand_dims(weights, 2)
    tmpZ = jnp.expand_dims(tmpZ, 1)
    zd = jnp.expand_dims(zd, 1)

    out = jnp.sum(weights * coeff[jnp.array([0, 2, 4, 6])], axis=0) * tmpZ + jnp.sum(
        weights * coeff[jnp.array([1, 3, 5, 7])], axis=0) * zd

    print(out.max())
    print(out.min())

    sigma = out[:, :1]
    rgb = out[:, 1:]

    sigma = jnp.clip(sigma, a_min=0.0, a_max=100000)
    rgb = rgb.reshape(-1, 3, 9)

    print(rgb.shape)
    print(sh.shape)
    sh_ray = sh[None, None, :]
    print(sh_ray.shape)
    rgb = rgb * sh_ray
    rgb_ = jnp.sum(rgb, axis=2)
    print(rgb_.shape)
    rgb_ = rgb_ + 0.5 #correction 1
    rgb_ = jnp.clip(rgb_, a_min=0.0, a_max=100000)
    tmp = step_size * sigma * delta_scale
    # tmp = np.clip(tmp, a_min=0.0, a_max=100000)
    var = 1 - jnp.exp(-tmp)
    Ti = jnp.exp(jnp.cumsum(-tmp))
    Ti = Ti[:, None]
    coefs = Ti * var
    rgb_ = coefs * rgb_
    rgb_ = jnp.sum(rgb_, axis=0)
    print(rgb_.shape)
    return rgb_


res = vmap(step1, in_axes=(0, 0, 0, 0, 0))(ori[:1000], dir[:1000], tmin[:1000], tmax[:1000], sh[:1000])

res = np.array(res)
samples_tmp = np.concatenate(samples_list)
samples_tmp = samples_tmp.reshape([1000, nb, 3])
np.testing.assert_allclose(res, samples_tmp)

weights_tmp = np.concatenate(weights_list)
weights_tmp = weights_tmp.reshape([1000, 4, nb])
np.testing.assert_allclose(res, weights_tmp)



batch_size = 4000
total_size = 800*800
res = []
i = 1
while i * batch_size < total_size:
    f = (i-1)*batch_size
    t = i*batch_size
    out = vmap(step1, in_axes=(0, 0, 0, 0, 0))(ori[f:t], dir[f:t], tmin[f:t], tmax[f:t], sh[f:t])
    print(out.max())
    print(out.min())
    res.append(out)
    i += 1
f = (i - 1) * batch_size

res.append(vmap(step1, in_axes=(0, 0, 0, 0, 0))(ori[f:], dir[f:], tmin[f:], tmax[f:], sh[f:]))
res = jnp.concatenate(res)
res = np.array(res)
img = res.reshape([800,800,3])

import cv2
# if nb_sh_channels == 2:
#     img = np.concatenate((img, np.zeros((img_size, img_size, 1)) + 0.5), axis=2)
img = (img * 255).astype(np.uint8)
# if nb_sh_channels == 3:
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# res = vmap(step1, in_axes=(0, 0, 0, 0))(ori, dir, tmin, sh)
tics = jnp.linspace(0, 1, 500)
tics = tics[:, None]
radius = 512
