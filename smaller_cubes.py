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

def create_sphere(radius, center):
    x = []
    y = []
    z = []
    for i in range(2000):
        u = np.random.normal(0, 1)
        v = np.random.normal(0, 1)
        w = np.random.normal(0, 1)
        norm = (u * u + v * v + w * w) ** (0.5)
        xi, yi, zi = u / norm, v / norm, w / norm
        x.append(xi)
        y.append(yi)
        z.append(zi)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    out = np.vstack([x, y, z]).T
    out *= radius
    out += center
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out)
    return pcd

def create_bbox(side=256):
    # Create the x, y, and z coordinates using meshgrid
    x, y, z = np.meshgrid(np.arange(side), 0, 0, indexing='ij')
    coords = np.stack([x, y, z], axis=-1)
    coords = coords.reshape([-1, 3])

    x, y, z = np.meshgrid(np.arange(side), side, 0, indexing='ij')
    coords_ = np.stack([x, y, z], axis=-1)
    coords_ = coords_.reshape([-1, 3])
    coords = np.vstack([coords, coords_])

    x, y, z = np.meshgrid(np.arange(side), 0, side, indexing='ij')
    coords_ = np.stack([x, y, z], axis=-1)
    coords_ = coords_.reshape([-1, 3])
    coords = np.vstack([coords, coords_])

    x, y, z = np.meshgrid(np.arange(side), side, side, indexing='ij')
    coords_ = np.stack([x, y, z], axis=-1)
    coords_ = coords_.reshape([-1, 3])
    coords = np.vstack([coords, coords_])

    x, y, z = np.meshgrid(0, np.arange(side), 0, indexing='ij')
    coords_ = np.stack([x, y, z], axis=-1)
    coords_ = coords_.reshape([-1, 3])
    coords = np.vstack([coords, coords_])

    x, y, z = np.meshgrid(side, np.arange(side), 0, indexing='ij')
    coords_ = np.stack([x, y, z], axis=-1)
    coords_ = coords_.reshape([-1, 3])
    coords = np.vstack([coords, coords_])

    x, y, z = np.meshgrid(0, np.arange(side), side, indexing='ij')
    coords_ = np.stack([x, y, z], axis=-1)
    coords_ = coords_.reshape([-1, 3])
    coords = np.vstack([coords, coords_])

    x, y, z = np.meshgrid(side, np.arange(side), side, indexing='ij')
    coords_ = np.stack([x, y, z], axis=-1)
    coords_ = coords_.reshape([-1, 3])
    coords = np.vstack([coords, coords_])

    x, y, z = np.meshgrid(0, 0, np.arange(side), indexing='ij')
    coords_ = np.stack([x, y, z], axis=-1)
    coords_ = coords_.reshape([-1, 3])
    coords = np.vstack([coords, coords_])

    x, y, z = np.meshgrid(side, 0, np.arange(side), indexing='ij')
    coords_ = np.stack([x, y, z], axis=-1)
    coords_ = coords_.reshape([-1, 3])
    coords = np.vstack([coords, coords_])

    x, y, z = np.meshgrid(0, side, np.arange(side), indexing='ij')
    coords_ = np.stack([x, y, z], axis=-1)
    coords_ = coords_.reshape([-1, 3])
    coords = np.vstack([coords, coords_])

    x, y, z = np.meshgrid(side, side, np.arange(side), indexing='ij')
    coords_ = np.stack([x, y, z], axis=-1)
    coords_ = coords_.reshape([-1, 3])
    coords = np.vstack([coords, coords_])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    return pcd

# def mipmap(mat):
#     assert mat.ndim == 3
#     xdim, ydim, zdim = mat.shape
#     assert xdim == ydim
#     assert ydim == zdim
#
#     levels = jnp.log2(xdim).astype(int)
#     mipmap = []
#
#     data = jnp.array(mat.astype(jnp.float32))
#
#     occupancy = jnp.array(data.astype(jnp.float32))
#     mipmap.append(occupancy.astype(int))
#
#     data = data[None, :, :, :, None]
#     kernel = jnp.ones([2, 2, 2])[:, :, :, jnp.newaxis, jnp.newaxis]
#     dn = lax.conv_dimension_numbers(data.shape, kernel.shape, ('NHWDC', 'HWDIO', 'NHWDC'))
#
#     for i in range(levels):
#         out = lax.conv_general_dilated(data,  # lhs = image tensor
#                                        kernel,  # rhs = conv kernel tensor
#                                        (2, 2, 2),  # window strides
#                                        'SAME',  # padding mode
#                                        (1, 1, 1),  # lhs/image dilation
#                                        (1, 1, 1),  # rhs/kernel dilation
#                                        dn)  # dimension_numbers
#
#         occupancy = out > 0
#         occupancy = jnp.array(occupancy.astype(jnp.float32))
#         data = occupancy
#         mipmap.append(occupancy[0, :, :, :, 0].astype(int))
#
#     return mipmap

def filter_over_sphere(grid, center, radius):
    xn, yn, zn = grid.shape
    x, y, z = np.meshgrid(np.arange(xn), np.arange(yn), np.arange(zn), indexing='ij')
    x = x - center[0]
    y = y - center[1]
    z = z - center[2]
    norm = x**2 + y**2 + z**2
    mask = norm < radius**2
    return mask



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
o3d.visualization.draw_geometries([bbox, pcd])

# Mipmap
mat = density_matrix
xdim, ydim, zdim = mat.shape

levels = jnp.log2(xdim).astype(int)
mipmap = []

mat = np.zeros(density_matrix.shape)
mat[density_matrix > 0.1] = 1
data = jnp.array(mat.astype(jnp.float32))

occupancy = jnp.array(data.astype(jnp.float32))
data = data[None, :, :, :, None]
kernel_list = []
dn_list = []
for i in range(levels):
    kernel = jnp.ones([2**i, 2**i, 2**i])[:, :, :, jnp.newaxis, jnp.newaxis]
    dn = lax.conv_dimension_numbers(data.shape, kernel.shape, ('NHWDC', 'HWDIO', 'NHWDC'))
    kernel_list.append(kernel)
    dn_list.append(dn)

# dn = lax.conv_dimension_numbers(data.shape, kernel.shape, ('NHWDC', 'HWDIO', 'NHWDC'))

for i in range(levels):
    dn = dn_list[i]
    kernel = kernel_list[i]
    out = lax.conv_general_dilated(data,  # lhs = image tensor
                                   kernel,  # rhs = conv kernel tensor
                                   (2**i, 2**i, 2**i),  # window strides
                                   'SAME',  # padding mode
                                   (1, 1, 1),  # lhs/image dilation
                                   (1, 1, 1),  # rhs/kernel dilation
                                   dn)  # dimension_numbers
    mipmap.append(out[0, :, :, :, 0])
    print(i)


# coords = np.indices((n, n, n))
# coords = coords[:, mipmap[0] > 0]
# coords = coords.reshape(3, -1).T
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(coords)
# o3d.visualization.draw_geometries([bbox, pcd])
#
# n = 128
# coords = np.indices((n, n, n))
# coords = coords[:, mipmap[1] > 0]
# coords = coords.reshape(3, -1).T
# coords = coords * 2
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(coords)
# o3d.visualization.draw_geometries([bbox, pcd])

# side = 256
# level = 0
# n = int(side / 2**level)
# coords = np.indices((n, n, n))
# coords = coords[:, mipmap[level] > 0]
# coords = coords.reshape(3, -1).T
# coords = coords * 2**level
# coords = coords + 2**level - 1
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(coords)

# pcd_list = []
# for level in range(8):
#     side = 256
#     n = int(side / 2 ** level)
#     coords = np.indices((n, n, n))
#     coords = coords[:, mipmap[level] > 0]
#     coords = coords.reshape(3, -1).T
#     coords = coords * 2 ** level
#     # coords = coords + 2 ** level - 1
#     pcd0 = o3d.geometry.PointCloud()
#     pcd0.points = o3d.utility.Vector3dVector(coords)
#     pcd_list.append(pcd0)

# it looks kinda correct, but unsure if this is what we want
# o3d.visualization.draw_geometries([bbox, pcd_list[7], pcd_list[6], pcd_list[5], pcd_list[4], pcd_list[3]])

# implement proper mipmap / octree visualization
# for a ray we want the first n voxels at the deepest level, and find that in the fastest way
# for a box, can compute which rays intersect it, with index?
# for a ray, can computer which boxes intersect it, with index?



import open3d as o3d
import open3d.visualization as vis
import numpy as np

total_width = 256
bbox = create_bbox(total_width)
level = 4
nb_bins = 256 / 2**level
assert nb_bins == len(mipmap[level])
box_width = 2**level

# box = o3d.geometry.TriangleMesh.create_box(box_width, box_width, box_width)
# box.compute_triangle_normals()

geoms = []
grid = mipmap[level]
for i in range(2**level):
    for j in range(2 ** level):
        for k in range(2 ** level):
            if grid[i, j, k] != 0:
                box = o3d.geometry.TriangleMesh.create_box(box_width, box_width, box_width)
                box.translate(np.array([box_width*i, box_width*j, box_width*k]))
                box.compute_triangle_normals()
                occ = grid[i, j, k] / box_width**3
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

geoms.append({'name': 'bbox', 'geometry': bbox})

vis.draw(geoms)