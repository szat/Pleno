import copy
import time

import jax.lax
import numpy as np
from trilinear_interpolation import *
from sampling_branch import intersect_ray_aabb
from spherical_harmonics import eval_sh_bases_mine
import open3d as o3d
from copy import deepcopy
import jax.numpy as jnp
from jax import lax
import numpy as np
from jax import grad, jit, vmap
from jax_mipmap import *

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

density_matrix = np.squeeze(npy_density_data[npy_links.clip(min=0)])
density_matrix[density_matrix < 0.2] = 0
mask_sphere = filter_over_sphere(density_matrix, np.ones(3)*128, 100)
density_matrix[~mask_sphere] = 0

n = 256
bbox = create_bbox(n)
coords = np.indices((n, n, n))
coords = coords[:, density_matrix > 0]
coords = coords.reshape(3, -1).T
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coords)
# o3d.visualization.draw_geometries([bbox, pcd])

occupancy_voxels = create_occupancy_voxels(density_matrix)

# Mipmap
mipmap = mipmap_compute(occupancy_voxels)
mipmap.reverse()

def viz_mipmap(mipmap, level, total_width):
    import open3d as o3d
    import open3d.visualization as vis
    import numpy as np

    nb_bins = 2**level
    assert nb_bins == len(mipmap[level])
    box_width = 256 / (2**level)

    box = o3d.geometry.TriangleMesh.create_box(box_width, box_width, box_width)
    box.compute_triangle_normals()

    geoms = []
    grid = mipmap[level]
    for i in range(2**level):
        for j in range(2 ** level):
            for k in range(2 ** level):
                if grid[i, j, k] != 0:
                    box = o3d.geometry.TriangleMesh.create_box(box_width, box_width, box_width)
                    box.translate(np.array([box_width*i, box_width*j, box_width*k]))
                    box.compute_triangle_normals()
                    occ = grid[i, j, k] / box_width**4
                    mat_box = vis.rendering.MaterialRecord()
                    mat_box.shader = 'defaultLitTransparency'
                    mat_box.base_color = [0.5, 0.5, 0.5, occ]
                    mat_box.base_roughness = 0.0
                    mat_box.base_reflectance = 0.0
                    mat_box.base_clearcoat = 1.0
                    mat_box.thickness = 1.0
                    mat_box.transmission = 1.0
                    mat_box.absorption_distance = 10
                    mat_box.absorption_color = [0.5, 0.5, 0.5]
                    name = 'box_'+str(i)+'_'+str(j)+'_'+str(k)
                    geoms.append({'name':name, 'geometry':box, 'material':mat_box})
    return geoms

def viz_create_box(side):
    coords = np.array([[0,0,0],
                       [0,0,side],
                       [0,side,side],
                       [0,side, 0],
                       [side, 0, 0],
                       [side, 0, side],
                       [side, side, side],
                       [side, side, 0]])
    pts = o3d.utility.Vector3dVector(coords)
    idx = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0],
                                      [4, 5], [5, 6], [6, 7], [7, 4],
                                      [0, 4], [1, 5], [2, 6], [3, 7]])
    lineset = o3d.geometry.LineSet(pts, idx)
    geoms = []
    name = "box_size_{}".format(str(side))
    geoms.append({'name': name, 'geometry': lineset})
    return geoms

def viz_lines(ori, end):
    if ori.ndim == 1:
        ori = ori[None, :]
        end = end[None, :]
    pts = o3d.utility.Vector3dVector(np.vstack([ori, end]))
    idx = []
    tmp = len(ori)
    for i in range(tmp):
        idx.append([0+i, tmp+i])
    idx = o3d.utility.Vector2iVector(idx)
    lineset = o3d.geometry.LineSet(pts, idx)
    name = "lineset_size{}".format(str(tmp))
    geoms = []
    geoms.append({'name': name, 'geometry': lineset})
    return geoms

ori = np.array([256, 256, 256])/2 - np.array([0, 0, 800])
x = np.linspace(0, 255, 800)
y = np.linspace(0, 255, 800)
# full coordinate arrays
xx, yy = np.meshgrid(x, y)
xx = xx.reshape(800*800)
yy = yy.reshape(800*800)
zz = np.ones(800*800)*256
end = np.vstack([xx, yy, zz]).transpose()
ori = np.tile(ori, [len(end), 1])

dir = end - ori
norm = np.linalg.norm(dir, axis=1)
norm = norm[:, None]
dir = dir / norm
inv = 1/dir
box_top = np.ones(3)*256
box_bottom = np.zeros(3)
sh = eval_sh_bases_mine(dir)
tmin, tmax = intersect_ray_aabb(ori, inv, box_bottom-0.5, box_top-0.5) # to compute the tmin and tmax, they do a offset of -0.5 first, so add -0.5 to box_min and box_max

mask = tmin < tmax
ori = ori[mask]
dir = dir[mask]
sh = sh[mask]
tmin = tmin[mask]
tmax = tmax[mask]

# very slow render
# step_size = 0.5
# delta_scale = 1/256
# colors = np.zeros([800*800, 3])
# max_dt = np.max(tmax - tmin)
# nb = 500
#
# res_non = []
# for i in range(len(ori)):
#     tics = np.linspace(tmin[i], max_dt + tmin[i], num=nb, dtype=np.float64)
#     samples = ori[i, None] + tics[:, None] * dir[i, None]
#     samples = np.clip(samples, 0, 254)
#     interp = trilinear_interpolation_shuffle_zero(samples, npy_links, npy_data)
#     # res_non.append(interp)
#
#     sigma = interp[:, :1]
#     rgb = interp[:, 1:]
#
#     sigma = np.clip(sigma, a_min=0.0, a_max=100000)
#     rgb = rgb.reshape(-1, 3, 9)
#
#     sh_ray = sh[i][None, None, :]
#     rgb = rgb * sh_ray
#
#     rgb = np.sum(rgb, axis=2)
#     rgb = rgb + 0.5 #correction 1
#     rgb = np.clip(rgb, a_min=0.0, a_max=100000)
#     tmp = step_size * sigma * delta_scale
#
#     var = 1 - np.exp(-tmp)
#     Ti = np.exp(np.cumsum(-tmp))
#     Ti = Ti[:, None]
#     coefs = Ti * var
#     rgb = coefs * rgb
#     rgb = np.sum(rgb, axis=0)
#
#     res_non.append(rgb)
# res_non = np.stack(res_non, axis=0)
#
# img = res_non.reshape([800,800,3])
# import cv2
# # if nb_sh_channels == 2:
# #     img = np.concatenate((img, np.zeros((img_size, img_size, 1)) + 0.5), axis=2)
# img = (img * 255).astype(np.uint8)
# # if nb_sh_channels == 3:
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

level = 4
# visualize mipmap / occupancy
geom_mipmap = viz_mipmap(mipmap, level, 256)
geom_bbox = viz_create_box(256)
# intersect one ray with mipmap / visualize
geom_rays = viz_lines(ori[24*14789], end[24*14789])

import open3d.visualization as vis
vis.draw(geom_mipmap+geom_bbox+geom_rays)

# for a set of samples of that ray, visualize all boolean flag
# for a set of samples of that ray, visualize the minimum
# find like 10 rays that intersect the object, do this for all of them

tmp_ori = ori[24*14789]
tmp_end = end[24*14789]
tmp_dir = dir[24*14789]
tmp_tmin = tmin[24*14789]
tmp_tmax = tmax[24*14789]

from jax.config import config
config.update("jax_enable_x64", True)

level = 4
occupancy = mipmap[level]
occupancy = occupancy != 0
# occupancy[occupancy != 0] = 1

nb = 100
tics = jnp.linspace(tmp_tmin, tmp_tmax, num=nb, dtype=jnp.float64)
samples = tmp_ori[None, :] + tics[:, None] * tmp_dir[None, :]
samples = jnp.clip(samples, 0, 254)

LARGE_NB = 10000000
def point_check_to_vmap(pt, occupancy):
    box_width = 256 / (2 ** level)
    tmp = pt / box_width
    tmp = tmp.astype(int)
    status = occupancy[tmp[0], tmp[1], tmp[2]].astype(int)
    tmp = pt * status
    # will be 0 if grid is not occupied (no live voxels)
    tmp = tmp + (1-status)*LARGE_NB # some large number
    # will be 10000 if grid is not occupied
    # value of pt otherwise
    return tmp

point_check = vmap(point_check_to_vmap, in_axes=(0, None))

res = point_check(samples, occupancy)
mask = res == LARGE_NB
colors = np.array([1, 0, 0])
colors = np.tile(colors, [len(samples), 1])
colors[mask] = 0
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(samples)
pcd.colors = o3d.utility.Vector3dVector(colors)
geom_pcd = [{'name':'pcd', 'geometry':pcd}]
vis.draw(geom_mipmap+geom_bbox+geom_pcd)

norm = jax.numpy.linalg.norm(res, axis=1)
min_idx = jax.numpy.argmin(norm)
min_norm = jax.numpy.min(norm)

colors = np.array([1, 1, 1])
colors = np.tile(colors, [len(samples), 1])
colors[min_idx] = np.array([0,0,1])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(samples)
pcd.colors = o3d.utility.Vector3dVector(colors)
geom_pcd = [{'name':'pcd', 'geometry':pcd}]
vis.draw(geom_mipmap+geom_bbox+geom_pcd)
for i in range(len(samples)):
    print(samples[i] / box_width)
    # print(occupancy[samples[i, 0].astype(int), samples[i, 1].astype(int), samples[i, 2].astype(int)])

# i = 1000
# ori_ = ori[i]
# dir_ = dir[i]
# tmin_ = tmin[i]
# tics = jnp.linspace(tmin_, max_dt + tmin_, num=nb, dtype=jnp.float64)
# samples_ = ori_[None, :] + tics[:, None] * dir_[None, :]
# samples_ = jnp.clip(samples_, 0, 254)
# filtered_samples_ = point_check(samples_, occupancy)
# filtered_samples_ = filtered_samples_ - ori_
# norm_ = jax.numpy.linalg.norm(filtered_samples_, axis=1)

def silhouette_to_vmap(ori, dir, tmin, tmax):
    tics = jnp.linspace(tmin, tmax, num=nb, dtype=jnp.float64)
    samples = ori[None, :] + tics[:, None] * dir[None, :]
    samples = jnp.clip(samples, 0, 254)
    filtered_samples = point_check(samples, occupancy)
    # filtered_samples = filtered_samples - ori
    norm = jax.numpy.linalg.norm(filtered_samples, axis=1)
    min_idx = jax.numpy.argmin(norm)
    min_norm = jax.numpy.min(norm)
    return samples[min_idx], min_norm

silhouette = jit(vmap(silhouette_to_vmap, in_axes=(0, 0, 0, 0)))

t0 = time.time()
res, norm = silhouette(ori, dir, tmin, tmax)
res.block_until_ready()
print(time.time() - t0)

mask = norm < 1500000
# visualize this

res = res[mask]

colors = np.array([0, 0, 1])
colors = np.tile(colors, [len(res), 1])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(res)
pcd.colors = o3d.utility.Vector3dVector(colors)
geom_pcd = [{'name':'pcd', 'geometry':pcd}]
vis.draw(geom_mipmap+geom_bbox+geom_pcd)



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


# tmp_rgb = []
for i in range(int(batch_nb - 1)):
    # i = 0
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        t0= time.time()
        res = jit_main(ori[i * batch_size: (i + 1) * batch_size],
                       dir[i * batch_size: (i + 1) * batch_size],
                       tmin[i * batch_size: (i + 1) * batch_size],
                       sh[i * batch_size: (i + 1) * batch_size],
                       max_dt, npy_links, npy_data)
        res.block_until_ready()
        print(time.time() - t0)
    # tmp_rgb.append(res)

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
# if nb_sh_channels == 2:
#     img = np.concatenate((img, np.zeros((img_size, img_size, 1)) + 0.5), axis=2)
img = (img * 255).astype(np.uint8)
# if nb_sh_channels == 3:
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


t0 = time.time()
res = jit_main(ori[:5000], dir[:5000], tmin[:5000], sh[:5000], max_dt, npy_links, npy_data)
print(time.time() - t0)

#
#
# inter_samples_jax = []
# for i in range(800*800, 100):
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
#     # sigma = interp[:, :1]
#     # rgb = interp[:, 1:]
#     #
#     # sigma = np.clip(sigma, a_min=0.0, a_max=100000)
#     # rgb = rgb.reshape(-1, 3, 9)
#     #
#     # sh_ray = sh[i][None, None, :]
#     # rgb = rgb * sh_ray
#     # rgb = np.sum(rgb, axis=2)
#     # rgb = rgb + 0.5 #correction 1
#     # rgb = np.clip(rgb, a_min=0.0, a_max=100000)
#     # tmp = step_size * sigma * delta_scale
#     # # tmp = np.clip(tmp, a_min=0.0, a_max=100000)
#     # var = 1 - np.exp(-tmp)
#     # Ti = np.exp(np.cumsum(-tmp))
#     # Ti = Ti[:, None]
#     # coefs = Ti * var
#     # rgb = coefs * rgb
#     # rgb = np.sum(rgb, axis=0)
#     # colors[i] = rgb
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
