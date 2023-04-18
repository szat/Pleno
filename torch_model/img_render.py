import copy
import os
import sys
sys.path.append('./torch_model/')

import cv2
import numpy as np
import open3d as o3d
import torch

import torch_model.model as model


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

######################## Rendering with pytorch model ####################################

path_to_weigths = "/home/diego/data/nerf/ckpt_syn/256_to_512_fasttv/lego/ckpt.npz"
img_size = 800
batch_size = 1024
nb_samples = 512

rf = model.RadianceField(idim=512, nb_samples=nb_samples)
data = np.load(path_to_weigths, allow_pickle=True)

# Access data arrays using keys
npy_radius = data['radius']
npy_center = data['center']
npy_links = data['links']
npy_density_data = data['density_data']
npy_sh_data = data['sh_data']
npy_basis_type = data['basis_type']

density_matrix = torch.from_numpy(np.squeeze(npy_density_data[npy_links.clip(min=0)]))
density_matrix[density_matrix < 0] = 0 # clip neg. density values
rf.opacity.data = density_matrix

# load the scene now: for a camera, get the rays
origin = np.array([513, 513, 513])
orientation = np.array([-1, -1, -1])
orientation = orientation / np.linalg.norm(orientation)
camera = Camera(origin=origin, orientation=orientation, dist_plane=1, length_x=1, length_y=1,
                pixels_x=img_size, pixels_y=img_size)

rays_cam = get_camera_rays(camera)
rays_origins = torch.from_numpy(np.tile(origin, (img_size*img_size, 1)))
rays_dirs = torch.from_numpy(rays_cam.reshape((img_size*img_size, 3)))

print("rendering!")
print(rays_origins.shape)
img_rgb = []
with torch.no_grad():
    for channel in range(3):
        color_batched = []
        print("channel:", channel + 1)
        sh_matrix = torch.from_numpy(npy_sh_data[:,channel*9:(channel + 1)*9][npy_links.clip(min=0)])
        rf.grid.data = sh_matrix
        for batch_start in range(0, rays_dirs.shape[0], batch_size):
            print("batching:", batch_start)
            origins_batched = rays_origins[batch_start:min(batch_start + batch_size, rays_dirs.shape[0])]
            dirs_batched = rays_dirs[batch_start:min(batch_start + batch_size, rays_dirs.shape[0])]
            color_batched.append(rf(origins_batched, dirs_batched))

        color_rays = torch.cat(color_batched)
        img_rgb.append(torch.reshape(color_rays, (img_size, img_size)))

img_rgb = torch.permute(torch.stack(img_rgb), (1, 2, 0))
img = img_rgb.detach().numpy()
img = (img * 255).astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(f"render2_lego_rgb_{img_size}x{img_size}.png", img)

