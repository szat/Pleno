import numpy as np
import os
import open3d as o3d
import copy

import numpy as np
import open3d as o3d
# data = np.load('/home/adrian/Documents/Nerf/256_to_512_fasttv/chair/ckpt.npz', allow_pickle=True)
data = np.load('/home/adrian/Code/svox2/opt/ckpt/exp2/ckpt.npz', allow_pickle=True)

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


def get_arrow(start_point, end_point, color=[0.3, 0.3, 0.3], thickness=1):
    vec = end_point - start_point
    norm = np.linalg.norm(vec)
    cone_height = norm * 0.2
    cylinder_height = norm * 0.8
    cone_radius = 0.2 * thickness
    cylinder_radius = 0.1 * thickness
    arrow = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
                                                     cone_height=cone_height,
                                                     cylinder_radius=cylinder_radius,
                                                     cylinder_height=cylinder_height)
    vec = vec / norm
    R = rotation_align(np.array([0, 0, 1]), vec)
    arrow.rotate(R, center=np.zeros(3))
    arrow.translate(start_point)
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color(color)
    return arrow


# Create the x, y, and z coordinates using meshgrid
x, y, z = np.meshgrid(np.arange(512), 0, 0, indexing='ij')
coords = np.stack([x, y, z], axis=-1)
coords = coords.reshape([-1, 3])

x, y, z = np.meshgrid(np.arange(512), 512, 0, indexing='ij')
coords_ = np.stack([x, y, z], axis=-1)
coords_ = coords_.reshape([-1, 3])
coords = np.vstack([coords, coords_])

x, y, z = np.meshgrid(np.arange(512), 0, 512, indexing='ij')
coords_ = np.stack([x, y, z], axis=-1)
coords_ = coords_.reshape([-1, 3])
coords = np.vstack([coords, coords_])

x, y, z = np.meshgrid(np.arange(512), 512, 512, indexing='ij')
coords_ = np.stack([x, y, z], axis=-1)
coords_ = coords_.reshape([-1, 3])
coords = np.vstack([coords, coords_])

x, y, z = np.meshgrid(0, np.arange(512), 0, indexing='ij')
coords_ = np.stack([x, y, z], axis=-1)
coords_ = coords_.reshape([-1, 3])
coords = np.vstack([coords, coords_])

x, y, z = np.meshgrid(512, np.arange(512), 0, indexing='ij')
coords_ = np.stack([x, y, z], axis=-1)
coords_ = coords_.reshape([-1, 3])
coords = np.vstack([coords, coords_])

x, y, z = np.meshgrid(0, np.arange(512), 512, indexing='ij')
coords_ = np.stack([x, y, z], axis=-1)
coords_ = coords_.reshape([-1, 3])
coords = np.vstack([coords, coords_])

x, y, z = np.meshgrid(512, np.arange(512), 512, indexing='ij')
coords_ = np.stack([x, y, z], axis=-1)
coords_ = coords_.reshape([-1, 3])
coords = np.vstack([coords, coords_])

x, y, z = np.meshgrid(0, 0, np.arange(512), indexing='ij')
coords_ = np.stack([x, y, z], axis=-1)
coords_ = coords_.reshape([-1, 3])
coords = np.vstack([coords, coords_])

x, y, z = np.meshgrid(512, 0, np.arange(512), indexing='ij')
coords_ = np.stack([x, y, z], axis=-1)
coords_ = coords_.reshape([-1, 3])
coords = np.vstack([coords, coords_])

x, y, z = np.meshgrid(0, 512, np.arange(512), indexing='ij')
coords_ = np.stack([x, y, z], axis=-1)
coords_ = coords_.reshape([-1, 3])
coords = np.vstack([coords, coords_])

x, y, z = np.meshgrid(512, 512, np.arange(512), indexing='ij')
coords_ = np.stack([x, y, z], axis=-1)
coords_ = coords_.reshape([-1, 3])
coords = np.vstack([coords, coords_])

# plane
x, y, z = np.meshgrid(0, np.arange(512), np.arange(512), indexing='ij')
coords_ = np.stack([x, y, z], axis=-1)
coords_ = coords_.reshape([-1, 3])
coords = np.vstack([coords, coords_])
coords = coords[::7,:]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coords)
# pcd.colors = o3d.utility.Vector3dVector(colors2)
o3d.visualization.draw_geometries([pcd])

# initial points for rays
x, y, z = np.meshgrid(-50, np.arange(510)+0.45, np.arange(510)+0.45, indexing='ij')
ray_origins = np.stack([x, y, z], axis=-1)
ray_origins = ray_origins.reshape([-1, 3])

ray_orientations = np.array([1, 0, 0])
ray_orientations = np.tile(ray_orientations, (len(ray_origins), 1))

r_list = []
for i in range(0, ray_origins.shape[0], 987):
    arrow = get_arrow(ray_origins[i, :], ray_origins[i, :] + 10 * ray_orientations[i,:], thickness=3)  # green
    r_list.append(arrow)

o3d.visualization.draw_geometries([pcd]+r_list)



import numpy as np
import open3d as o3d
# data = np.load('/home/adrian/Documents/Nerf/256_to_512_fasttv/chair/ckpt.npz', allow_pickle=True)
data = np.load('/home/adrian/Code/svox2/opt/ckpt/exp2/ckpt.npz', allow_pickle=True)

# Access data arrays using keys
npy_radius = data['radius']
npy_center = data['center']
npy_links = data['links']
# npy_density_data2 = data['density_data']
npy_density_data = data['density_data'].astype(dtype=np.float16)
npy_sh_data = data['sh_data'].astype(dtype=np.float16)
npy_basis_type = data['basis_type']

# this is not correct
density_matrix = np.squeeze(npy_density_data[npy_links.clip(min=0)])
sh_matrix = np.squeeze(npy_sh_data[npy_links.clip(min=0)])
sh_matrix = sh_matrix[:,:,:,[0, 9, 19]]

n = 256
coords = np.indices((n, n, n)).reshape(3, -1).T


# Create a numpy array of colors
opacities = density_matrix.flatten().reshape(-1, 1)

mask_pos = np.squeeze(opacities > 0)
# mask_neg = np.squeeze(opacities < 0)

coords_pos = coords[mask_pos,:]
# coords_neg = coords[mask_neg,:]

sh_matrix = sh_matrix.flatten().reshape(3,-1).T
sh_matrix_pos = sh_matrix[mask_pos, :]

#evaluate the spherical harmonics
coef = 0.5*np.sqrt(1/np.pi)
sh_matrix_pos = sh_matrix_pos*coef
sh_matrix_pos = sh_matrix_pos - np.min(sh_matrix_pos)
sh_matrix_pos = sh_matrix_pos/np.max(sh_matrix_pos)
# sh_matrix_pos = sh_matrix_pos.astype(dtype=float)
#
# opacities_pos = opacities[mask_pos,:]
# tmp = np.exp(-opacities_pos)
# tmp = 1 - tmp
#
# coords_sub = coords_pos[:, :]
# colors_sub = tmp[:, :].astype(dtype=float)
# colors_sub = np.hstack([colors_sub, colors_sub, colors_sub])
# colors_sub = np.ones_like(colors_sub)

# Create an Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coords_pos)
pcd.colors = o3d.utility.Vector3dVector(sh_matrix_pos)
# pcd.compute_vertex_normals()
o3d.visualization.draw_geometries([pcd])


#now do for color


# get samples for each ray
from sampling_branch import *
rays = np.hstack([ray_origins, ray_orientations])
samples_list = rays_to_samples(rays[0:10, :], 0.37, np.ones(3)*512, np.zeros(3))
samples = np.stack(samples_list).astype(dtype=np.float16)

# get trilinear interpolation for each ray
coeffs_list = samples_to_interpolation_coeffs(samples_list)
coeffs = np.stack(coeffs_list).astype(dtype=np.float16)








#
# import numpy as np
# import torch
# torch.cuda.is_available()
# torch.cuda.device_count()
# import sys
# sys.path.append('/home/adrian/Code/Pleno/torch_model')
# sys.path.append('/home/adrian/Code/Pleno/torch_model/')
# import torch_model.model as model
#
# rf = model.RadianceField(idim=512, nb_samples=512)

import numpy as np
import open3d as o3d
data = np.load('/home/adrian/Documents/Nerf/256_to_512_fasttv/chair/ckpt.npz', allow_pickle=True)

# Access data arrays using keys
npy_radius = data['radius']
npy_center = data['center']
npy_links = data['links']
npy_density_data = data['density_data']
npy_sh_data = data['sh_data']
npy_basis_type = data['basis_type']

npy_density_data = npy_density_data - np.min(npy_density_data)
npy_density_data = npy_density_data / np.max(npy_density_data)

density_matrix = np.squeeze(npy_density_data[npy_links.clip(min=0)])
n = 512
M = density_matrix
# M = np.random.rand(n, n, n)

# Flatten the numpy array and create a numpy array of coordinates
coords = np.indices((n, n, n)).reshape(3, -1).T

# Create a numpy array of colors
colors = M.flatten().reshape(-1, 1)

threshold = 0.41
valid_points_mask = M.flatten() >= threshold
coords = coords[valid_points_mask]
colors = colors[valid_points_mask]
colors2 = np.hstack([colors, colors, colors])

coords = coords[::10, :]
colors2 = colors2[::10, :]

# Create an Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coords)
pcd.colors = o3d.utility.Vector3dVector(colors2)

o3d.visualization.draw_geometries([pcd])

pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

import matplotlib.pyplot as plt
plt.hist(npy_density_data)
plt.savefig("mygraph.png")
plt.close()

# use pointcloud instead

voxel = o3d.geometry.TriangleMesh.create_box(5, 5, 5)
voxel.compute_vertex_normals()

b_list = []
for i in range(0, 512, 4):
    for j in range(0, 512, 4):
        for k in range(0, 512, 4):
            opacity = density_matrix[i, j, k]
            new_voxel = copy.deepcopy(voxel)
            new_voxel.translate([i, j, k])
            voxel.paint_uniform_color([opacity, 0, opacity])
            b_list.append(new_voxel)





o3d.visualization.draw_geometries([bbox, coord_w, coord_bbox, cam_z, cam_x, cam_y]+r_list+b_list)


# density_matrix = np.empty([512, 512, 512])
# sh_matrix = np.empty([512, 512, 512, 9])
# density_matrix = torch.from_numpy(np.squeeze(npy_density_data[npy_links.clip(min=0)]))
# sh_matrix = torch.from_numpy(npy_sh_data[:,:9][npy_links.clip(min=0)])
# rf.grid = torch.nn.Parameter(sh_matrix)
# rf.opacity = torch.nn.Parameter(density_matrix)

# load the scene now
# for a camera, get the rays

ray_orientations = np.tile(origin, (50 * 50, 1))
rrr = rays.reshape((50*50, 3))
res = rf.forward(torch.from_numpy(ray_orientations), torch.from_numpy(rrr))

