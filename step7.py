import numpy as np
from trilinear_interpolation import *
from sampling_branch import intersect_ray_aabb
from spherical_harmonics import eval_sh_bases_mine
import jax.numpy as jnp
from jax import lax
import numpy as np
from jax import grad, jit, vmap
from jax.config import config
import jax
import jax.numpy as jnp
import time

config.update("jax_enable_x64", True)

path = '/home/adrian/Code/svox2/opt/ckpt/exp2/ckpt.npz'
img_size = 800
# batch_size = 4*1024
# nb_samples = 512
# nb_sh_channels = 3
data = np.load(path, allow_pickle=True)
npy_radius = data['radius']
npy_center = data['center']
npy_links = data['links']
npy_links = npy_links[::2, ::2, ::2]
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
nb = 100

# ori = ori[::1000, :]
# dir = dir[::1000, :]
# tmin = tmin[::1000]

def trilinear_interpolation_to_vmap(vecs, links, values_compressed):
    # xyz = vecs - origin
    # xyz = xyz / delta_voxel
    xyz = vecs
    xyz_floor = jnp.floor(xyz)
    xd, yd, zd = xyz - xyz_floor
    x0, y0, z0 = xyz_floor.astype(int)

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
    # return coeff, weights, tmpZ, zd
    weights = weights[:, None]

    out = jnp.sum(weights * coeff[[0, 2, 4, 6], :], axis=0) * tmpZ + jnp.sum(weights * coeff[[1, 3, 5, 7], :], axis=0) * zd
    out = out[None, :]
    return out
jit_interp = jit(vmap(trilinear_interpolation_to_vmap, in_axes=(0, None, None)))

def main_to_vmap(ori, dir, tmin, sh, max_dt, npy_links, npy_data):
    tics = jnp.linspace(tmin, max_dt + tmin, num=nb, dtype=jnp.float64)
    samples = ori[None, :] + tics[:, None] * dir[None, :]
    samples = jnp.clip(samples, 0, 254)
    interp = jit_interp(samples, npy_links, npy_data)
    interp = jnp.squeeze(interp)

    sigma = interp[:, :1]
    rgb = interp[:, 1:]

    sigma = jnp.clip(sigma, a_min=0.0, a_max=100000)
    rgb = rgb.reshape(-1, 3, 9)

    sh_ray = sh[None, None, :]
    rgb = rgb * sh_ray

    rgb = jnp.sum(rgb, axis=2)
    rgb = rgb + 0.5 #correction 1
    rgb = jnp.clip(rgb, a_min=0.0, a_max=100000)
    tmp = step_size * sigma * delta_scale

    var = 1 - jnp.exp(-tmp)
    Ti = jnp.exp(jnp.cumsum(-tmp))
    Ti = Ti[:, None]
    coefs = Ti * var
    rgb = coefs * rgb
    rgb = jnp.sum(rgb, axis=0)
    return rgb
jit_main = jit(vmap(main_to_vmap, in_axes=(0, 0, 0, 0, None, None, None)))

# res = jit_main(ori, dir, tmin, sh, max_dt, npy_links, npy_data)
# res = jnp.squeeze(res)
#
# np.testing.assert_almost_equal(res, res_non) # from step6.py

ori = jax.device_put(jnp.array(ori))
dir = jax.device_put(jnp.array(dir))
tmin = jax.device_put(jnp.array(tmin))
npy_links = jax.device_put(jnp.array(npy_links))
npy_data = jax.device_put(jnp.array(npy_data))

batch_size = 5000
batch_nb = jnp.ceil(len(ori) / batch_size)

tmp_rgb2 = jnp.zeros([800*800, 3])


# tmp_rgb = []
# for i in range(int(batch_nb - 1)):
#     t0= time.time()
#     tmp_rgb2.at[i * batch_size: (i + 1) * batch_size, :].set(jit_main(ori[i * batch_size: (i + 1) * batch_size],
#                    dir[i * batch_size: (i + 1) * batch_size],
#                    tmin[i * batch_size: (i + 1) * batch_size],
#                    sh[i * batch_size: (i + 1) * batch_size],
#                    max_dt, npy_links, npy_data))
#     print(time.time() - t0)
#     # tmp_rgb.append(res)


tmp_rgb = []
for i in range(int(batch_nb - 1)):
    res = jit_main(ori[i * batch_size: (i + 1) * batch_size],
                   dir[i * batch_size: (i + 1) * batch_size],
                   tmin[i * batch_size: (i + 1) * batch_size],
                   sh[i * batch_size: (i + 1) * batch_size],
                   max_dt, npy_links, npy_data)
    res.block_until_ready()
    tmp_rgb.append(res)

last_dab = len(ori) - (batch_nb - 1) * batch_size
res = jit_main(ori[int((batch_nb - 1) * batch_size):],
               dir[int((batch_nb - 1) * batch_size):],
               tmin[int((batch_nb - 1) * batch_size):],
               sh[int((batch_nb - 1) * batch_size):],
               max_dt, npy_links, npy_data)
tmp_rgb.append(res)
colors = np.concatenate(tmp_rgb)

img = colors.reshape([800,800,3])
import cv2
img = (img * 255).astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)




# t0 = time.time()
# res = jit_main(ori[:5000], dir[:5000], tmin[:5000], sh[:5000], max_dt, npy_links, npy_data)
# print(time.time() - t0)
#
# #
# #
# # inter_samples_jax = []
# for i in range(800*800):
#     # x = max_dt + tmin[i] - tmax[i]
#     tics = jnp.linspace(tmin[i], max_dt + tmin[i], num=nb)
#     samples = ori[i, None] + tics[:, None] * dir[i, None]
#     samples = jnp.clip(samples, 0, 254)
#     samples = jnp.array(samples)
#     interp = jit_interp(samples, npy_links, npy_data, jnp.zeros(3), jnp.ones(3))
#
#     # inter_samples_jax.append(interp)
#     # for i in range(len(inter_samples_jax)):
#     #     np.testing.assert_almost_equal(inter_samples_non[i], inter_samples_jax[i])
#
#     sigma = interp[:, :1]
#     rgb = interp[:, 1:]
#
#     sigma = np.clip(sigma, a_min=0.0, a_max=100000)
#     rgb = rgb.reshape(-1, 3, 9)
#
#     sh_ray = sh[i][None, None, :]
#     rgb = rgb * sh_ray
#     rgb = np.sum(rgb, axis=2)
#     rgb = rgb + 0.5 #correction 1
#     rgb = np.clip(rgb, a_min=0.0, a_max=100000)
#     tmp = step_size * sigma * delta_scale
#     # tmp = np.clip(tmp, a_min=0.0, a_max=100000)
#     var = 1 - np.exp(-tmp)
#     Ti = np.exp(np.cumsum(-tmp))
#     Ti = Ti[:, None]
#     coefs = Ti * var
#     rgb = coefs * rgb
#     rgb = np.sum(rgb, axis=0)
#     colors[i] = rgbnpy_links
#
# img = colors.reshape([800,800,3])
# import cv2
# # if nb_sh_channels == 2:
# #     img = np.concatenate((img, np.zeros((img_size, img_size, 1)) + 0.5), axis=2)
# img = (img * 255).astype(np.uint8)
# # if nb_sh_channels == 3:
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import jax.numpy as jnp
# from jax import grad, jit, vmap
# from jax import random
# import numpy as np
# from jax import device_put
# import jax
# import time
# import contextlib
# import timeit
#
# from time import time
#
# from sampling_branch import intersect_ray_aabb
# from spherical_harmonics import eval_sh_bases_mine
# from trilinear_interpolation import trilinear_jax
# # def timer_func(func):
# #     import numpy
# #     def wrap_func(*args, **kwargs):
# #         times = np.zeros(30)
# #         for i in range(30):
# #             t1 = time()
# #             result = func(*args, **kwargs)
# #             t2 = time()
# #             times[i] = t2-t1
# #         print(f'Function {func.__name__!r} executed in {times.mean()*1000:.4f}ms, std: {times.std()*1000:.4f}ms')
# #         return result
# #     return wrap_func
# #
# # #define random link grid with integer values going from negative to n.
# #
# # nb_links = 16*16*16
# # nb_non_neg = 1000
# # links_idx = np.arange(nb_links)
# # idx_in_links = np.random.choice(links_idx, size=nb_non_neg, replace=False)
# # values_idx = np.arange(nb_non_neg).astype(int)
# # links = np.ones([16, 16, 16]) * -1
# # links = links.reshape(-1)
# # links[idx_in_links] = values_idx
# # links = links.reshape([16, 16, 16])
# # links = jnp.int32(links)
# # values_compressed = np.random.rand(nb_non_neg, 2)
# # vecs = np.random.rand(100, 3)
# # vecs = vecs * 15
# #
# # # now we have all of our values, lets send them all to the gpu via jax and see what happens
# # links = jnp.array(links)
# # values_compressed = jnp.array(values_compressed)
# # vecs = jnp.array(vecs)
# # origin=jnp.zeros(3)
# # delta_voxel=jnp.ones(3)
# #
# # # print(links.device_buffer.device()) (vecs)
# # xyz_floor = jnp.floor(vecs)
# # diff = vecs - xyz_floor
# # x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]
# # x0 = jnp.int32(x0)
# # y0 = jnp.int32(y0)
# # z0 = jnp.int32(z0)
# # l000 = jnp.int32(links[x0, y0, z0])
# # v000 = values_compressed[l000]
#
#
# def trilinear_interpolation_shuffle_zero(vecs, links, values_compressed, origin=np.zeros(3), delta_voxel=np.ones(3)):
#     if vecs.ndim == 1:
#         vecs = np.expand_dims(vecs, 0)
#     if values_compressed.ndim == 1:
#         values_compressed = np.expand_dims(values_compressed, 1)
#
#     vecs = vecs - origin
#     vecs = vecs / delta_voxel
#     xyz_floor = np.floor(vecs)
#     diff = vecs - xyz_floor
#     xd, yd, zd = diff[:, 0], diff[:, 1], diff[:, 2]
#     xyz_floor = np.int32(xyz_floor)
#     x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]
#
#     l000 = links[x0, y0, z0]
#     l100 = links[x0+1, y0, z0]
#     l010 = links[x0, y0+1, z0]
#     l001 = links[x0, y0, z0+1]
#     l110 = links[x0+1, y0+1, z0]
#     l011 = links[x0, y0+1, z0+1]
#     l101 = links[x0+1, y0, z0+1]
#     l111 = links[x0+1, y0+1, z0+1]
#
#     v000 = values_compressed[l000]
#     v100 = values_compressed[l100]
#     v010 = values_compressed[l010]
#     v001 = values_compressed[l001]
#     v110 = values_compressed[l110]
#     v011 = values_compressed[l011]
#     v101 = values_compressed[l101]
#     v111 = values_compressed[l111]
#
#     tmpX = 1 - xd
#     tmpY = 1 - yd
#     tmpZ = 1 - zd
#     a000 = tmpX * tmpY
#     a100 = xd * tmpY
#     a010 = tmpX * yd
#     a110 = xd * yd
#
#     weights = np.array([a000, a010, a100, a110])
#     coeff = np.array([v000, v001, v010, v011, v100, v101, v110, v111])
#     weights = np.expand_dims(weights, 2)
#     tmpZ = np.expand_dims(tmpZ, 1)
#     zd = np.expand_dims(zd, 1)
#
#     out = np.sum(weights * coeff[np.array([0, 2, 4, 6])], axis=0) * tmpZ + np.sum(weights * coeff[np.array([1, 3, 5, 7])], axis=0) * zd
#     return out
#
# path = '/home/adrian/Code/svox2/opt/ckpt/exp2/ckpt.npz'
# img_size = 800
# # batch_size = 4*1024
# # nb_samples = 512
# # nb_sh_channels = 3
# data = np.load(path, allow_pickle=True)
# npy_radius = data['radius']
# npy_center = data['center']
# npy_links = data['links']
# npy_density_data = data['density_data']
# npy_sh_data = data['sh_data']
# npy_basis_type = data['basis_type']
#
# # hack
# npy_density_data[0] = 0
# npy_sh_data[0] = 0
# npy_links[npy_links < 0] = 0
# npy_data = np.hstack([npy_density_data, npy_sh_data])
#
# npy_links = jnp.array(npy_links)
# npy_data = jnp.array(npy_data)
#
# ori = np.load("/home/adrian/Documents/temp/svox_ori.npy")
# dir = np.load("/home/adrian/Documents/temp/svox_dir.npy")
#
# # transform to world coords
# step_size = 0.5
# delta_scale = 1/256
# gsz = 256
# offset = 0.5 * gsz - 0.5  # 0.5 * 256 - 0.5
# scaling = 0.5 * gsz  # 0.5 * 256
# ori = offset + ori * scaling
#
# # get the tic samples
# spacing = 0.5
# box_top = np.ones(3)*256
# box_bottom = np.zeros(3)
# dir = dir / np.expand_dims(np.linalg.norm(dir, axis=1), 1)
# inv = 1/dir
# sh = eval_sh_bases_mine(dir)
# tmin, tmax = intersect_ray_aabb(ori, inv, box_bottom-0.5, box_top-0.5) # to compute the tmin and tmax, they do a offset of -0.5 first, so add -0.5 to box_min and box_max
#
# mask = tmin < tmax
# ori = ori[mask]
# dir = dir[mask]
# sh = sh[mask]
# tmin = tmin[mask]
# tmax = tmax[mask]
# max_dt = np.max(tmax - tmin)
# nb = 700# len(np.arange(tmin[0], max_dt + tmin[0], spacing))
# tics = np.zeros([len(tmin), nb])
# for i in range(len(tmin)):
#     tics[i] = np.linspace(tmin[i], max_dt + tmin[i], num=nb)
# colors = jnp.zeros([800*800, 3])
#
# np.testing.assert_almost_equal(tics, tics2)
#
# tmin = tmin[:1000]
# def fct(tmin):
#     tics = jnp.linspace(tmin, max_dt + tmin, num=nb)
#
#
#     return res
#
#
# v_fct = jit(vmap(fct, in_axes=(0)))
# res = v_fct(tmin)
#
# np.testing.assert_almost_equal(res, tics)
#
#
#
# sh = jnp.array(sh)
# ori = jnp.array(ori)
# dir = jnp.array(dir)
# tmin = jnp.array(tmin)
# tmax = jnp.array(tmax)
#
# colors = jnp.zeros([800*800, 3])
# tmin = tmin[:, jnp.newaxis]
# inputs = jnp.hstack([ori, dir, tmin, sh])
# # def get_color(ori, dir, tmin, sh):
#
# @jit
# def get_color(inputs):
#     ori = inputs[:3]
#     dir = inputs[3:6]
#     tmin = inputs[6:7]
#     sh = inputs[7:]
#     ori = ori.reshape([1, 3])
#     dir = dir.reshape([1, 3])
#     tics = jnp.arange(tmin, max_dt + tmin, spacing)
#     # tics = tics[:, jnp.newaxis]
#     # samples = ori + tics * dir
#     # samples = jnp.clip(samples, 0, 254)
#     # interp = trilinear_interpolation_shuffle_zero(samples, npy_links, npy_data)
#     # sigma = interp[:, :1]
#     # rgb = interp[:, 1:]
#     #
#     # sigma = jnp.clip(sigma, a_min=0.0, a_max=100000)
#     # rgb = rgb.reshape(-1, 3, 9)
#     # sh = sh[jnp.newaxis, jnp.newaxis, :]
#     # tmp_rgb = jnp.clip(jnp.sum(rgb * sh, axis=2) + 0.5, a_min=0.0, a_max=100000)
#     # tmp = spacing * sigma * delta_scale
#     # var = 1 - jnp.exp(-tmp)
#     # Ti = jnp.exp(jnp.cumsum(-tmp))
#     # Ti = Ti[:, jnp.newaxis]
#     # coefs = Ti * var
#     # out = jnp.sum(coefs * tmp_rgb, axis=0)
#     # return out
#     return tics
#
# def vmap_get_color(v_ori, v_dir, v_tmin, v_sh):
#     return vmap(get_color)(v_ori, v_dir, v_tmin, v_sh)
#
# res = vmap_get_color(ori, dir, tmin, sh)
#
#
# for i in range(800*800):
#     # i = 65000
#     samples = ori[i, None] + tics[i][:, None] * dir[i, None]
#     samples = np.clip(samples, 0, 254)
#     interp = trilinear_interpolation_shuffle_zero(samples, npy_links, npy_data)
#     sigma = interp[:, :1]
#     rgb = interp[:, 1:]
#
#     sigma = np.clip(sigma, a_min=0.0, a_max=100000)
#     rgb = rgb.reshape(-1, 3, 9)
#
#     sh_ray = sh[i][None, None, :]
#     rgb = rgb * sh_ray
#     rgb = np.sum(rgb, axis=2)
#     rgb = rgb + 0.5 #correction 1
#     rgb = np.clip(rgb, a_min=0.0, a_max=100000)
#     tmp = step_size * sigma * delta_scale
#     # tmp = np.clip(tmp, a_min=0.0, a_max=100000)
#     var = 1 - np.exp(-tmp)
#     Ti = np.exp(np.cumsum(-tmp))
#     Ti = Ti[:, None]
#     coefs = Ti * var
#     rgb = coefs * rgb
#     rgb = np.sum(rgb, axis=0)
#     colors[i] = rgb
#     print(i)
#
# img = colors.reshape([800,800,3])
# img = np.array(img)
# import cv2
# # if nb_sh_channels == 2:
# #     img = np.concatenate((img, np.zeros((img_size, img_size, 1)) + 0.5), axis=2)
# img = (img * 255).astype(np.uint8)
# # if nb_sh_channels == 3:
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
