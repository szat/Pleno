import copy
import os
import sys
sys.path.append('.')
from tqdm import tqdm


import numpy as np
# import open3d as o3d
import torch

import model
# from torch_model import *
# from torch_model import model
from sampling_branch import intersect_ray_aabb
from spherical_harmonics import sh_cartesian, eval_sh_bases_mine

# https://iquilezles.org/articles/noacos/
def rotation_align(from_vec, to_vec):
    assert from_vec.shape == to_vec.shape, "from_vec and to_vec need to be of the same shape"
    if from_vec.ndim == 1:
        v = np.cross(from_vec, to_vec)
        # c = np.einsum('ij,ij...->i...', from_vec, to_vec)
        c = np.dot(from_vec, to_vec)
        if np.all(v == np.zeros(3)) and c > 0:
            return np.eye(3)
        if np.all(v == np.zeros(3)) and c < 0:
            return -np.eye(3)
        k = 1.0 / (1.0 + c)
        return np.array([[v[0]**2 * k + c,    v[0]*v[1]*k - v[2], v[0]*v[2]*k + v[1]],
                         [v[0]*v[1]*k + v[2], v[1]**2 * k + c,    v[1]*v[2]*k - v[0]],
                         [v[0]*v[2]*k - v[1], v[1]*v[2]*k + v[0], v[2]**2 * k + c   ]])
    if from_vec.ndim == 2:
        v = np.cross(from_vec, to_vec)
        c = np.einsum('ij,ij...->i...', from_vec, to_vec)
        k = 1.0 / (1.0 + c)
        out = np.array([[v[:, 0]**2 * k + c,    v[:, 0]*v[:, 1]*k - v[:, 2], v[:, 0]*v[:, 2]*k + v[:, 1]],
                         [v[:, 0]*v[:, 1]*k + v[:, 2], v[:, 1]**2 * k + c,    v[:, 1]*v[:, 2]*k - v[:, 0]],
                         [v[:, 0]*v[:, 2]*k - v[:, 1], v[:, 1]*v[:, 2]*k + v[:, 0], v[:, 2]**2 * k + c   ]])
        out = np.einsum('ijk->kij', out) # rearrange dimensions
        bool_flag_identity = np.all(v == np.zeros(3), axis=1) * c > 0
        bool_flag_reverse = np.all(v == np.zeros(3), axis=1) * c < 0
        out[bool_flag_identity] = np.eye(3)
        out[bool_flag_reverse] = -np.eye(3)
        return out


class Camera:
    def __init__(self,
                 origin=np.zeros(3),
                 orientation=np.array([0, 0, 1]),
                 dist_plane=1,
                 length_x=0.640,
                 length_y=0.480,
                 pixels_x=640,
                 pixels_y=480):
        self.origin = origin
        self.orientation = orientation
        self.dist_plane = dist_plane
        self.length_x = length_x
        self.length_y = length_y
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y


# def get_arrow(start_point, end_point, color=[0.3, 0.3, 0.3], thickness=1):
#     vec = end_point - start_point
#     norm = np.linalg.norm(vec)
#     cone_height = norm * 0.2
#     cylinder_height = norm * 0.8
#     cone_radius = 0.2 * thickness
#     cylinder_radius = 0.1 * thickness
#     arrow = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
#                                                      cone_height=cone_height,
#                                                      cylinder_radius=cylinder_radius,
#                                                      cylinder_height=cylinder_height)
#     vec = vec / norm
#     R = rotation_align(np.array([0, 0, 1]), vec)
#     arrow.rotate(R, center=np.zeros(3))
#     arrow.translate(start_point)
#     arrow.compute_vertex_normals()
#     arrow.paint_uniform_color(color)
#     return arrow


def get_camera_vectors(camera: Camera):
    z = np.array([0, 0, 1])
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])

    # Put into world coordinates
    R = rotation_align(z, camera.orientation)
    z = np.dot(R, z)
    x = np.dot(R, x)
    y = np.dot(R, y)

    # Camera template
    z = z * camera.dist_plane
    x = x * camera.length_x
    y = y * camera.length_y

    # z = z + camera.origin
    # x = x + camera.origin
    # y = y + camera.origin
    return z, x, y

def get_camera_rays(camera: Camera):
    z, x, y = get_camera_vectors(camera)
    tics_x = np.expand_dims(np.linspace(-1, 1, camera.pixels_x), 1)
    tics_y = np.expand_dims(np.linspace(-1, 1, camera.pixels_y), 1)

    xx = tics_x * x
    yy = tics_y * y
    xx = np.expand_dims(xx, 0)
    yy = np.expand_dims(yy, 1)
    rays = xx + yy
    zz = np.expand_dims(z, [0, 1])
    rays = rays + zz
    rays = rays / np.expand_dims(np.linalg.norm(rays, axis=2), 2)
    return rays


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

# opacity = npy_density_data[npy_links.clip(min=0)]
# # opacity[opacity < 0] = 0.0
# opacity = np.expand_dims(opacity, 3)
# grid = npy_sh_data[:, :9][npy_links.clip(min=0)]


tmp = npy_links.clip(min=0)
mask = npy_links >= 0
tmp2 = np.zeros_like(npy_links)
tmp2[mask] = npy_links[mask]
np.testing.assert_allclose(tmp, tmp2)
# same

grid = npy_sh_data[npy_links.clip(min=0)]
grid2 = np.zeros_like(grid)
mask = npy_links >= 0
grid2[mask] = npy_sh_data[npy_links[mask]]
np.testing.assert_allclose(grid, grid2)
#not same
grid = np.zeros([256, 256, 256, 27])
grid[mask] = npy_sh_data[npy_links[mask]]

opacity = npy_density_data[npy_links.clip(min=0)]
opacity = np.zeros([256, 256, 256, 1])
opacity[mask] = npy_density_data[npy_links[mask]]
# opacity[opacity < 0] = 0.0
# opacity = np.expand_dims(opacity, 3)


origin = np.array([257., 257., 257.])
orientation = np.array([-1, -1, -1])
orientation = orientation / np.linalg.norm(orientation)
camera = Camera(origin=origin, orientation=orientation, dist_plane=1, length_x=1, length_y=1,
                pixels_x=img_size, pixels_y=img_size)



rays_cam = get_camera_rays(camera)
rays_ori = np.tile(origin, (img_size*img_size, 1))
rays_dir = rays_cam.reshape((img_size*img_size, 3))


spacing = 0.5
box_top = np.ones(3)*256
box_bottom = np.zeros(3)

# from svox.py line 1494
gsz = 256
offset = 0.5 * gsz - 0.5  # 0.5 * 256 - 0.5
scaling = 0.5 * gsz  # 0.5 * 256
# offset + points * scaling
# load data of rays
ori_svox = np.load("/home/adrian/Documents/temp/ori.npy")
dir_svox = np.load("/home/adrian/Documents/temp/dir.npy")
ori_svox = offset + ori_svox * scaling

from sampling_branch import intersect_ray_aabb

# to compute the tmin and tmax, they do a offset of -0.5 first, so add -0.5 to box_min and box_max
rays_dir = dir_svox
rays_dir = rays_dir / np.expand_dims(np.linalg.norm(rays_dir, axis=1), 1)
rays_inv = 1/rays_dir
rays_ori = ori_svox
tmin_, tmax_ = intersect_ray_aabb(rays_ori, rays_inv, box_bottom-0.5, box_top-0.5)
mask = tmin_ < tmax_
rays_ori = rays_ori[mask]
rays_dir = rays_dir[mask]
tmin_ = tmin_[mask]
tmax_ = tmax_[mask]
tics = []
for i in range(len(tmin_)):
    tics.append(np.arange(tmin_[i], tmax_[i], spacing))

# rays_dir = rays_dir / np.expand_dims(np.linalg.norm(rays_dir, axis=1), 1)
# rays_inv = 1/rays_dir
# tmin, tmax = intersect_ray_aabb(rays_ori, rays_inv, box_bottom, box_top)
# mask = tmin < tmax
# rays_ori = rays_ori[mask]
# rays_dir = rays_dir[mask]
# tmin = tmin[mask]
# tmax = tmax[mask]
# tics = []
# for i in range(len(tmin)):
#     tics.append(np.arange(tmin[i], tmax[i], spacing))



# delta_ijk = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
#                                        (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)],
#                                        dtype=np.int64).reshape((8, 3))
#
# K_sh = np.array([0.28209479, 0.48860251, 0.48860251, 0.48860251, 1.09254843,
#                                  1.09254843, 0.31539157, 1.09254843, 0.54627422])
#
# delta_voxel = np.array([1, 1, 1], dtype=float)



# from trilinear_interpolation import *
#
# colors = []
# for i, t in enumerate(tics):
iter = 0
sample_points_5000 = np.zeros([5000, 3])
for i in range(5000):
    # t = tics[i]
    ori = rays_ori[i]
    dir = rays_dir[i]
    ori = np.expand_dims(ori, axis=1)
    dir = np.expand_dims(dir, axis=1)
    t = tics[i]
    samples = (ori + t[iter] * dir).T
    ori = ori.T
    dir = dir.T

    # frustum = samples.astype(int)
    # frustum = np.expand_dims(frustum, [0, 2])
    #
    # delta_ijk_ = np.expand_dims(delta_ijk, [0, 1])
    # frustum = frustum + delta_ijk_
    sample_points = samples
    sample_points = np.clip(sample_points, 0, 254) # svox2.py, line 717
    sample_points_5000[i] = sample_points

np.testing.assert_allclose(pos.numpy(), sample_points_5000)


pos_svox = np.load("/home/adrian/Documents/temp/pos_pre_trilinear_0_5000.npy")
l_svox = np.load("/home/adrian/Documents/temp/l_pre_trilinear_0_5000.npy")




xyz = sample_points_5000
# xyz = xyz - origin
# xyz = xyz / np.array([dx, dy, dz])  # , dtype=float), axis=1)

xyz_floor = np.floor(xyz)
# np.sum(np.abs(xyz_floor - l.numpy()))
diff = xyz - xyz_floor

xd, yd, zd = diff[:, 0], diff[:, 1], diff[:, 2]

xyz_floor = xyz_floor.astype(int)
x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]

x_zeros = np.zeros(x0.shape, dtype=int)
x_zeros[xd >= 0] += 1
x1 = x0 + x_zeros

y_zeros = np.zeros(y0.shape, dtype=int)
y_zeros[yd >= 0] += 1
y1 = y0 + y_zeros

z_zeros = np.zeros(z0.shape, dtype=int)
z_zeros[zd >= 0] += 1
z1 = z0 + z_zeros

np.testing.assert_allclose(grid[x0, y0, z0], rgb000.numpy())
np.testing.assert_allclose(grid[x1, y0, z0], rgb100.numpy())
np.testing.assert_allclose(grid[x0, y1, z0], rgb010.numpy())
np.testing.assert_allclose(grid[x1, y1, z0], rgb110.numpy())

np.testing.assert_allclose(grid[x0, y0, z1], rgb001.numpy())
np.testing.assert_allclose(grid[x1, y0, z1], rgb101.numpy())
np.testing.assert_allclose(grid[x0, y1, z1], rgb011.numpy())
np.testing.assert_allclose(grid[x1, y1, z1], rgb111.numpy())


from trilinear_interpolation import *

sh_mine = np.zeros([5000, 9])
interp_sh_coeffs_mine = np.zeros([5000, 27])
interp_opacities_mine = np.zeros([5000, 1])

interp_sh_coeffs = trilinear_interpolation(sample_points_5000, grid)
interp_opacities = trilinear_interpolation(sample_points_5000, opacity)

np.testing.assert_allclose(interp_sh_coeffs, rgb.numpy())
np.testing.assert_allclose(interp_opacities, sigma.numpy())



for i in range(5000):
    dir = rays_dir[i]
    dir = np.expand_dims(dir, 0)
    sh = eval_sh_bases_mine(dir)
    sh_mine[i] = sh
    sample_points = sample_points_5000[i]
    sample_points = np.expand_dims(sample_points, 0)
    interp_sh_coeffs = trilinear_interpolation(sample_points, grid)
    interp_sh_coeffs_mine[i] = interp_sh_coeffs
    interp_opacities = trilinear_interpolation(sample_points, opacity)
    interp_opacities_mine[i] = interp_opacities






    # sh = sh_cartesian(dir)
    sh = eval_sh_bases_mine(dir)
    interp_sh_coeffs = trilinear_interpolation(sample_points, grid)
    interp_opacities = trilinear_interpolation(sample_points, opacity)

    interp_opacities[interp_opacities < 0] = 0.0
    sigma = interp_opacities
    delta = spacing
    weights = 1 - np.exp(- sigma * delta)
    Ti = np.exp(-np.cumsum(sigma * delta))
    sh_channels = np.einsum("i...,i...,ij...->j...", Ti, weights, interp_sh_coeffs)
    r = np.sum(sh_channels[:9])
    g = np.sum(sh_channels[9:18])
    b = np.sum(sh_channels[18:])
    rgb = np.array([r, g, b])
    colors.append(rgb)

complete_colors = np.zeros((np.tile(origin, (img_size*img_size, 1)).shape[0], 3))
rendered_rays = np.array(colors)
# rendered_rays = rendered_rays + 0.5
# rendered_rays[rendered_rays < 0] = 0
complete_colors[mask] = rendered_rays
img = np.reshape(complete_colors, (img_size, img_size, 3))
img = np.array(img)

import cv2

# if nb_sh_channels == 2:
#     img = np.concatenate((img, np.zeros((img_size, img_size, 1)) + 0.5), axis=2)
img = (img * 255).astype(np.uint8)
# if nb_sh_channels == 3:
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(f"render.png", img)
