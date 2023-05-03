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

from sampling_branch import intersect_ray_aabb
from spherical_harmonics import eval_sh_bases_mine

# def timer_func(func):
#     import numpy
#     def wrap_func(*args, **kwargs):
#         times = np.zeros(30)
#         for i in range(30):
#             t1 = time()
#             result = func(*args, **kwargs)
#             t2 = time()
#             times[i] = t2-t1
#         print(f'Function {func.__name__!r} executed in {times.mean()*1000:.4f}ms, std: {times.std()*1000:.4f}ms')
#         return result
#     return wrap_func
#
# #define random link grid with integer values going from negative to n.
#
# nb_links = 16*16*16
# nb_non_neg = 1000
# links_idx = np.arange(nb_links)
# idx_in_links = np.random.choice(links_idx, size=nb_non_neg, replace=False)
# values_idx = np.arange(nb_non_neg).astype(int)
# links = np.ones([16, 16, 16]) * -1
# links = links.reshape(-1)
# links[idx_in_links] = values_idx
# links = links.reshape([16, 16, 16])
# links = jnp.int32(links)
# values_compressed = np.random.rand(nb_non_neg, 2)
# vecs = np.random.rand(100, 3)
# vecs = vecs * 15
#
# # now we have all of our values, lets send them all to the gpu via jax and see what happens
# links = jnp.array(links)
# values_compressed = jnp.array(values_compressed)
# vecs = jnp.array(vecs)
# origin=jnp.zeros(3)
# delta_voxel=jnp.ones(3)
#
# # print(links.device_buffer.device()) (vecs)
# xyz_floor = jnp.floor(vecs)
# diff = vecs - xyz_floor
# x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]
# x0 = jnp.int32(x0)
# y0 = jnp.int32(y0)
# z0 = jnp.int32(z0)
# l000 = jnp.int32(links[x0, y0, z0])
# v000 = values_compressed[l000]

@jit
def trilinear_interpolation_shuffle_zero(vecs, links, values_compressed, origin=jnp.zeros(3), delta_voxel=jnp.ones(3)):
    if vecs.ndim == 1:
        vecs = jnp.expand_dims(vecs, 0)
    if values_compressed.ndim == 1:
        values_compressed = jnp.expand_dims(values_compressed, 1)

    vecs = vecs - origin
    vecs = vecs / delta_voxel
    xyz_floor = jnp.floor(vecs)
    diff = vecs - xyz_floor
    xd, yd, zd = diff[:, 0], diff[:, 1], diff[:, 2]
    xyz_floor = jnp.int32(xyz_floor)
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
    coeff = jnp.array([v000, v001, v010, v011, v100, v101, v110, v111])
    weights = jnp.expand_dims(weights, 2)
    tmpZ = jnp.expand_dims(tmpZ, 1)
    zd = jnp.expand_dims(zd, 1)

    out = jnp.sum(weights * coeff[jnp.array([0, 2, 4, 6])], axis=0) * tmpZ + jnp.sum(weights * coeff[jnp.array([1, 3, 5, 7])], axis=0) * zd
    return out

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

npy_links = jnp.array(npy_links)
npy_data = jnp.array(npy_data)

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
max_dt = np.max(tmax-tmin)
tmp = np.arange(0, max_dt, spacing)
colors = np.zeros([800*800, 3])
nb = len(tmp)
tics = np.zeros([len(tmin), nb])
for i in range(len(tmin)):
    tmp = np.arange(tmin[i], tmax[i] + max_dt - (tmax[i] - tmin[i]), spacing)

    dt = dt / spacing
    tmp = np.arange(tmin[i], tmax[i], spacing)
    tics.append(np.arange(tmin[i], tmax[i], spacing))
    tics[i] = np.linspace(tmin[i], tmax[i], nb)

sh = jnp.array(sh)
ori = jnp.array(ori)
dir = jnp.array(dir)
tics = jnp.array(tics)

def get_color(ori, dir, tics):
    ori = ori.reshape([1, 3])
    dir = dir.reshape([1, 3])
    tics = tics.reshape([nb, 1])
    samples = ori + tics * dir
    samples = jnp.clip(samples, 0, 254)
    interp = trilinear_interpolation_shuffle_zero(samples, npy_links, npy_data)
    sigma = interp[:, :1]
    rgb = interp[:, 1:]

    sigma = jnp.clip(sigma, a_min=0.0, a_max=100000)
    rgb = rgb.reshape(-1, 3, 9)
    sh_ray = sh[i][None, None, :]
    tmp_rgb = jnp.clip(jnp.sum(rgb * sh_ray, axis=2) + 0.5, a_min=0.0, a_max=100000)
    tmp = spacing * sigma * delta_scale
    var = 1 - np.exp(-tmp)
    Ti = jnp.exp(jnp.cumsum(-tmp))
    Ti = Ti[:, None]
    coefs = Ti * var
    return jnp.sum(coefs * tmp_rgb, axis=0)

@jit
def vmap_get_color(v_ori, v_dir, v_tics):
    return vmap(get_color)(v_ori, v_dir, v_tics)




for i in range(800*800):
    # i = 65000
    samples = ori[i, None] + tics[i][:, None] * dir[i, None]
    samples = np.clip(samples, 0, 254)
    interp = trilinear_interpolation_shuffle_zero(samples, npy_links, npy_data)
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
