import copy
import os
import sys
sys.path.append('.')

import cv2
import numpy as np
import open3d as o3d
import torch

#from camera import Camera
from sampling_branch import intersect_ray_aabb 
import model



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

"""
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
"""
# load the scene now: for a camera, get the rays
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
