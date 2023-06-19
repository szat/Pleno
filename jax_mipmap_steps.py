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

def create_line(start, end, nb):
    x = np.linspace(start[0], end[0], nb)
    y = np.linspace(start[1], end[1], nb)
    z = np.linspace(start[2], end[2], nb)
    out = np.vstack([x, y, z]).T
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

def mipmap_compute(mat):
    xdim, ydim, zdim = mat.shape

    levels = jnp.log2(xdim).astype(int)# add 1 for the root node
    mipmap = []

    occupancy_mat = np.zeros(mat.shape)
    occupancy_mat[mat > 0] = 1
    nb_of_pts = occupancy_mat.sum()
    data = jnp.array(occupancy_mat.astype(jnp.float32))

    data = data[None, :, :, :, None]
    kernel_list = []
    dn_list = []
    for i in range(levels):
        kernel = jnp.ones([2**i, 2**i, 2**i])[:, :, :, jnp.newaxis, jnp.newaxis]
        dn = lax.conv_dimension_numbers(data.shape, kernel.shape, ('NHWDC', 'HWDIO', 'NHWDC'))
        kernel_list.append(kernel)
        dn_list.append(dn)

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
        print("Finished level i = " + str(i))

    mipmap.append(jnp.array([[[nb_of_pts]]]))
    return mipmap

def filter_over_sphere(grid, center, radius):
    xn, yn, zn = grid.shape
    x, y, z = np.meshgrid(np.arange(xn), np.arange(yn), np.arange(zn), indexing='ij')
    x = x - center[0]
    y = y - center[1]
    z = z - center[2]
    norm = x**2 + y**2 + z**2
    mask = norm < radius**2
    return mask

def viz_mipmap(mipmap, level, total_width, ray_ori = np.zeros(3), ray_end = np.ones(3)*256, touched = []):
    import open3d as o3d
    import open3d.visualization as vis
    import numpy as np

    bbox = create_bbox(total_width)
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

    for i in range(len(touched)):
        ori = touched[i]
        box = o3d.geometry.TriangleMesh.create_box(box_width, box_width, box_width)
        box.translate(ori * box_width)
        box.compute_triangle_normals()
        mat_box = vis.rendering.MaterialRecord()
        mat_box.shader = 'defaultLitTransparency'
        occ = 1
        mat_box.base_color = [1, 0, 0, occ]
        mat_box.base_roughness = 1.0
        mat_box.base_reflectance = 0.0
        mat_box.base_clearcoat = 1.0
        mat_box.thickness = 1.0
        mat_box.transmission = 0.5
        mat_box.absorption_distance = 10
        mat_box.absorption_color = [1, 0, 0]
        name = 'box_touch' + str(i)
        geoms.append({'name': name, 'geometry': box, 'material': mat_box})

    line = create_line(ray_ori, ray_end, nb_bins * 4)
    geoms.append({'name': 'bbox', 'geometry': bbox})
    geoms.append({'name': 'line', 'geometry': line})
    return geoms

def create_occupancy_voxels(density_matrix):
    mat = density_matrix
    occupancy_mat = np.zeros(mat.shape)
    occupancy_mat[mat > 0] = 1
    data = jnp.array(occupancy_mat.astype(jnp.float32))
    data = data[None, :, :, :, None]
    kernel = jnp.ones([2, 2, 2])[:, :, :, jnp.newaxis, jnp.newaxis]
    dn = lax.conv_dimension_numbers(data.shape, kernel.shape, ('NHWDC', 'HWDIO', 'NHWDC'))
    out = lax.conv_general_dilated(data,  kernel, (1, 1, 1), 'SAME', (1, 1, 1), (1, 1, 1), dn)
    out = out[0, :, :, :, 0]
    mask = np.array(out == 8)
    return mask.astype(int)

def check_table(table, next_table):
    shift = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
    for i in range(len(table)):
        for j in range(8):
            if table[i, j] != -1:
                xyz = table[i, -3:] * 2
                idx_next = int(table[i, j])
                xyz_next = next_table[idx_next, -3:]
                xyz = xyz + shift[j]
                np.testing.assert_almost_equal(xyz, xyz_next)
    return True

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
# o3d.visualization.draw_geometries([bbox, pcd])

occupancy_voxels = create_occupancy_voxels(density_matrix)

# Mipmap
mipmap = mipmap_compute(occupancy_voxels)
mipmap.reverse()


def make_table_last(grid):
    x, y, z = np.where(grid)
    idx = jnp.arange(0, grid.sum())
    tmp = np.ones_like(grid) * -1
    tmp[grid] = idx
    grid = tmp
    for i in range(len(x)):
        assert grid[x[i], y[i], z[i]] == i
    table = np.ones([len(x), 3]) * -1
    table[:, 0] = x
    table[:, 1] = y
    table[:, 2] = z
    # table[:,11] = level
    return table

def make_table(grid_coarse, grid_fine):
    grid = grid_fine
    x, y, z = np.where(grid)
    idx = jnp.arange(0, grid.sum())
    tmp = np.ones_like(grid) * -1
    tmp[grid] = idx
    grid = tmp
    for i in range(len(x)):
        assert grid[x[i], y[i], z[i]] == i

    grid_000 = grid[0::2, 0::2, 0::2]
    grid_001 = grid[0::2, 0::2, 1::2]
    grid_010 = grid[0::2, 1::2, 0::2]
    grid_100 = grid[1::2, 0::2, 0::2]
    grid_011 = grid[0::2, 1::2, 1::2]
    grid_110 = grid[1::2, 1::2, 0::2]
    grid_101 = grid[1::2, 0::2, 1::2]
    grid_111 = grid[1::2, 1::2, 1::2]

    mask_coarse = grid_coarse.flatten()
    f_000 = grid_000.flatten()
    f_001 = grid_001.flatten()
    f_010 = grid_010.flatten()
    f_100 = grid_100.flatten()
    f_011 = grid_011.flatten()
    f_110 = grid_110.flatten()
    f_101 = grid_101.flatten()
    f_111 = grid_111.flatten()

    f_000 = f_000[mask_coarse]
    f_001 = f_001[mask_coarse]
    f_010 = f_010[mask_coarse]
    f_100 = f_100[mask_coarse]
    f_011 = f_011[mask_coarse]
    f_110 = f_110[mask_coarse]
    f_101 = f_101[mask_coarse]
    f_111 = f_111[mask_coarse]

    table = np.ones([mask_coarse.sum(), 8+3]) * -1
    table[:, 0] = f_000
    table[:, 1] = f_001
    table[:, 2] = f_010
    table[:, 3] = f_100
    table[:, 4] = f_011
    table[:, 5] = f_110
    table[:, 6] = f_101
    table[:, 7] = f_111
    x, y, z = np.where(grid_coarse) # coarse
    table[:, 8] = x
    table[:, 9] = y
    table[:, 10] = z
    # table[:,11] = level
    return table

# def intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max):
#     # considers the boundary of the volume as NON intersecting, if tmax <= tmin then NO intersection
#     if ray_origin.ndim == 1:
#         ray_origin = np.expand_dims(ray_origin, 0)
#         ray_inv_dir = np.expand_dims(ray_inv_dir, 0)
#     tmin = np.ones(len(ray_origin)) * -np.inf
#     tmax = np.ones(len(ray_origin)) * np.inf
#     t0 = (box_min - ray_origin) * ray_inv_dir
#     t1 = (box_max - ray_origin) * ray_inv_dir
#     tsmaller = np.nanmin([t0, t1], axis=0)
#     tbigger = np.nanmax([t0, t1], axis=0)
#     tmin = np.max([tmin, np.max(tsmaller, axis=1)], axis=0)
#     tmax = np.min([tmax, np.min(tbigger, axis=1)], axis=0)
#     return tmin, tmax

g0 = mipmap[0] > 0
g1 = mipmap[1] > 0
g2 = mipmap[2] > 0
g3 = mipmap[3] > 0
g4 = mipmap[4] > 0
g5 = mipmap[5] > 0
g6 = mipmap[6] > 0
g7 = mipmap[7] > 0
g8 = mipmap[8] > 0

table0 = make_table(g0, g1)
table1 = make_table(g1, g2)
table2 = make_table(g2, g3)
table3 = make_table(g3, g4)
table4 = make_table(g4, g5)
table5 = make_table(g5, g6)
table6 = make_table(g6, g7)
table7 = make_table(g7, g8)
# table8 = make_table(g8, g8, 8)
table8 = make_table_last(g8)

tables_list = [table0, table1, table2, table3, table4, table5, table6, table7, table8]

check_table(table0, table1)
check_table(table1, table2)
check_table(table2, table3)
check_table(table3, table4)
check_table(table4, table5)
check_table(table5, table6)
check_table(table6, table7)
check_table(table7, table8)

# def search_mipmap_table(tables_list, rays):
ray_dir = np.array([1, 0.8, 0.8])
ray_dir = ray_dir / np.linalg.norm(ray_dir)
ray_inv_dir = 1/ray_dir
ray_ori = np.ones(3) * 0


def ray_mipmap_intersect(ray_ori, ray_inv_dir, tables_list, max_level):
    cube_touched_idx = [0]
    new_list = []
    cube_ori_list = []
    for level in range(max_level+1):
        table = tables_list[level]  # from mipmap 2 to mipmap 3
        cube_size = 256 / (2 ** level)
        print(len(cube_touched_idx))
        for i in cube_touched_idx:
            cube_ori = table[i, -3:] * cube_size
            cube_end = cube_ori + cube_size
            tn, tf = intersect_ray_aabb(ray_ori, ray_inv_dir, cube_ori, cube_end)
            if tn <= tf:
                if level == max_level:
                    cube_ori_list.append(table[i, -3:])
                    mask = table[i, :8] != -1
                    idx = table[i, :8][mask]
                    idx = idx.astype(int)
                    new_list += idx.tolist()
                else:
                    mask = table[i, :8] != -1
                    idx = table[i, :8][mask]
                    idx = idx.astype(int)
                    new_list += idx.tolist()
        cube_touched_idx = new_list
        new_list = []
    return cube_ori_list


def ray_mipmap_intersect3(ray_ori, ray_inv_dir, tables_list, max_level):
    cube_touched_idx = np.array([0])
    for level in range(max_level+1):
        table = tables_list[level]  # from mipmap 2 to mipmap 3
        cube_size = 256 / (2 ** level)
        tn = np.zeros_like(cube_touched_idx)
        tf = np.zeros_like(cube_touched_idx)
        cube_ori = table[cube_touched_idx, -3:] * cube_size
        cube_end = cube_ori + cube_size

        for it in range(len(cube_touched_idx)):
            tn[it], tf[it] = intersect_ray_aabb(ray_ori, ray_inv_dir, cube_ori[it], cube_end[it])

        mask = tn <= tf
        cube_touched_idx = cube_touched_idx[mask]
        table = table[cube_touched_idx]
        mask = table[:,:8] != -1
        idx = table[:, :8][mask]
        idx = idx.astype(int)
        new_list = idx

        if level == max_level:
            return table[:, -3:]

        cube_touched_idx = new_list
    return table[:, -3:]

# do all the lines and use jit?
# table is of fixed size per grid
def ray_grid_intersect(ray_ori, ray_inv_dir, table, level):
    cube_size = 256 / (2 ** level)
    idx_list = []
    for i in range(len(table)):
        cube_ori = table[i, -3:] * cube_size
        cube_end = cube_ori + cube_size
        tn, tf = intersect_ray_aabb(ray_ori, ray_inv_dir, cube_ori, cube_end)
        if tn <= tf:
            idx_list.append(i)
    return table[idx_list, -3:]

def check_in_grid(pts, grid):
    pts = np.array(pts)
    grid = np.array(grid).astype(int)
    for i in range(len(pts)):
        pt = pts[i]
        flag = grid[int(pt[0]), int(pt[1]), int(pt[2])]
        if flag != 1:
            print("not equal at ith={} position ({}, {}, {})".format(i, int(pt[0]), int(pt[1]), int(pt[2])))

sort_pt = np.array([0, 3, -10])
max_level = 5
table = make_table_last(g5)
out_a = ray_grid_intersect(ray_ori, ray_inv_dir, table, max_level)
out_a = np.array(out_a)
dist = np.linalg.norm(out_a - sort_pt, axis = 1)
out_a = out_a[dist.argsort()]

check_in_grid(out_a, g5)

out1 = ray_mipmap_intersect(ray_ori, ray_inv_dir, tables_list, max_level)
out1 = np.array(out1)
dist = np.linalg.norm(out1 - sort_pt, axis = 1)
out1 = out1[dist.argsort()]

check_in_grid(out1, g5)

out3 = ray_mipmap_intersect3(ray_ori, ray_inv_dir, tables_list, max_level) # incorrect
out3 = np.array(out3)
dist = np.linalg.norm(out3 - sort_pt, axis = 1)
out3 = out3[dist.argsort()]

check_in_grid(out3, g5)

np.testing.assert_almost_equal(np.array(out_a), np.array(out1))
np.testing.assert_almost_equal(np.array(out3), np.array(out1))
