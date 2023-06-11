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

def create_line(start, end):
    x = np.linspace(start[0], end[0], 100)
    y = np.linspace(start[1], end[1], 100)
    z = np.linspace(start[2], end[2], 100)
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

    for i in range(len(touched)):
        ori = cube_touched[i]
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
        name = 'box_touch' + str(i) + '_' + str(j) + '_' + str(k)
        geoms.append({'name': name, 'geometry': box, 'material': mat_box})

    line = create_line(ray_ori, ray_end)
    geoms.append({'name': 'bbox', 'geometry': bbox})
    geoms.append({'name': 'line', 'geometry': line})
    return geoms


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

# Mipmap
mipmap = mipmap_compute(density_matrix)
mipmap.reverse()

# Viz
# geoms = viz_mipmap(mipmap, 3, 256)
# import open3d as o3d
# import open3d.visualization as vis
# vis.draw(geoms)

def check_table(table, next_table):
    shift = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
    for i in range(len(table)):
        for j in range(8):
            if table[i, j] != -1:
                # table2[i, j]
                xyz = table[i, -3:] * 2
                idx_next = int(table[i, j])
                xyz_next = next_table[idx_next, -3:]
                xyz = xyz + shift[j]
                np.testing.assert_almost_equal(xyz, xyz_next)
    print("Alles gut!")
    return

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
    return table


def intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max):
    # considers the boundary of the volume as NON intersecting, if tmax <= tmin then NO intersection
    if ray_origin.ndim == 1:
        ray_origin = np.expand_dims(ray_origin, 0)
        ray_inv_dir = np.expand_dims(ray_inv_dir, 0)
    tmin = np.ones(len(ray_origin)) * -np.inf
    tmax = np.ones(len(ray_origin)) * np.inf
    t0 = (box_min - ray_origin) * ray_inv_dir
    t1 = (box_max - ray_origin) * ray_inv_dir
    tsmaller = np.nanmin([t0, t1], axis=0)
    tbigger = np.nanmax([t0, t1], axis=0)
    tmin = np.max([tmin, np.max(tsmaller, axis=1)], axis=0)
    tmax = np.min([tmax, np.min(tbigger, axis=1)], axis=0)
    return tmin, tmax


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
tables_list = [table0, table1, table2, table3, table4, table5, table6, table7]

check_table(table0, table1)
check_table(table1, table2)
check_table(table2, table3)
check_table(table3, table4)
check_table(table4, table5)
check_table(table5, table6)
check_table(table6, table7)

# def search_mipmap_table(tables_list, rays):
ray_dir = np.ones(3)/np.sqrt(3)
ray_inv_div = 1/ray_dir
ray_ori = np.ones(3) * 0

# select the cubes that you want to intersect test (all of them?)
level = 3
table = tables_list[level] # from mipmap 2 to mipmap 3
cube_list = np.arange(0, len(table)) # to start
cube_touched = []
cube_size = 256/(2**level)
for i in cube_list:
    cube_ori = table[i, 8:] * cube_size
    cube_end = cube_ori + cube_size - 1
    tn, tf = intersect_ray_aabb(ray_ori, ray_inv_div, cube_ori, cube_end)
    # print("tmin = {}, tmax = {}".format(tn, tf))
    if tn <= tf:
        cube_touched.append(np.array(table[i, 8:]))
        print("cube {} intersects the ray".format(i))


geoms = viz_mipmap(mipmap, level, 256, np.zeros(3), np.ones(3)*256, cube_touched)
import open3d as o3d
import open3d.visualization as vis
vis.draw(geoms)




# grid = g3
# x3, y3, z3 = np.where(grid)
# idx = jnp.arange(0, grid.sum())
# tmp = np.ones_like(grid) * -1
# tmp[grid] = idx
# tmp2 = tmp.flatten()
# tmp3 = tmp2 != -1
# tmp4 = tmp2[tmp3]
# grid = tmp
# for i in range(len(x3)):
#     assert grid[x3[i], y3[i], z3[i]] == i
#
# grid_000 = grid[0::2, 0::2, 0::2]
# grid_001 = grid[0::2, 0::2, 1::2]
# grid_010 = grid[0::2, 1::2, 0::2]
# grid_100 = grid[1::2, 0::2, 0::2]
# grid_011 = grid[0::2, 1::2, 1::2]
# grid_110 = grid[1::2, 1::2, 0::2]
# grid_101 = grid[1::2, 0::2, 1::2]
# grid_111 = grid[1::2, 1::2, 1::2]
#
# mask_coarse = g2.flatten()
# f_000 = grid_000.flatten()
# f_001 = grid_001.flatten()
# f_010 = grid_010.flatten()
# f_100 = grid_100.flatten()
# f_011 = grid_011.flatten()
# f_110 = grid_110.flatten()
# f_101 = grid_101.flatten()
# f_111 = grid_111.flatten()
#
# f_000 = f_000[mask_coarse]
# f_001 = f_001[mask_coarse]
# f_010 = f_010[mask_coarse]
# f_100 = f_100[mask_coarse]
# f_011 = f_011[mask_coarse]
# f_110 = f_110[mask_coarse]
# f_101 = f_101[mask_coarse]
# f_111 = f_111[mask_coarse]
#
# table2 = np.ones([mask_coarse.sum(), 8+3]) * -1
# table2[:, 0] = f_000
# table2[:, 1] = f_001
# table2[:, 2] = f_010
# table2[:, 3] = f_100
# table2[:, 4] = f_011
# table2[:, 5] = f_110
# table2[:, 6] = f_101
# table2[:, 7] = f_111
# x2, y2, z2 = np.where(g2) # coarse
# table2[:, 8] = x2
# table2[:, 9] = y2
# table2[:, 10] = z2
#
# # next
# grid = g4
# x4, y4, z4 = np.where(grid)
# idx = jnp.arange(0, grid.sum())
# tmp = np.ones_like(grid) * -1
# tmp[grid] = idx
# tmp2 = tmp.flatten()
# tmp3 = tmp2 != -1
# tmp4 = tmp2[tmp3]
# grid = tmp
# for i in range(len(x3)):
#     assert grid[x4[i], y4[i], z4[i]] == i
#
# grid_000 = grid[0::2, 0::2, 0::2]
# grid_001 = grid[0::2, 0::2, 1::2]
# grid_010 = grid[0::2, 1::2, 0::2]
# grid_100 = grid[1::2, 0::2, 0::2]
# grid_011 = grid[0::2, 1::2, 1::2]
# grid_110 = grid[1::2, 1::2, 0::2]
# grid_101 = grid[1::2, 0::2, 1::2]
# grid_111 = grid[1::2, 1::2, 1::2]
#
# mask_coarse = g3.flatten()
# f_000 = grid_000.flatten()
# f_001 = grid_001.flatten()
# f_010 = grid_010.flatten()
# f_100 = grid_100.flatten()
# f_011 = grid_011.flatten()
# f_110 = grid_110.flatten()
# f_101 = grid_101.flatten()
# f_111 = grid_111.flatten()
#
# f_000 = f_000[mask_coarse]
# f_001 = f_001[mask_coarse]
# f_010 = f_010[mask_coarse]
# f_100 = f_100[mask_coarse]
# f_011 = f_011[mask_coarse]
# f_110 = f_110[mask_coarse]
# f_101 = f_101[mask_coarse]
# f_111 = f_111[mask_coarse]
#
# table3 = np.ones([mask_coarse.sum(), 8+3]) * -1
# table3[:, 0] = f_000
# table3[:, 1] = f_001
# table3[:, 2] = f_010
# table3[:, 3] = f_100
# table3[:, 4] = f_011
# table3[:, 5] = f_110
# table3[:, 6] = f_101
# table3[:, 7] = f_111
# x3_, y3_, z3_ = np.where(g3) # coarse
# table3[:, 8] = x3_
# table3[:, 9] = y3_
# table3[:, 10] = z3_
#
# table0_ = make_table(g0, g1)
# table1_ = make_table(g1, g2)
# table2_ = make_table(g2, g3)
# table3_ = make_table(g3, g4)
#
# check_table(table0_, table1_)
# check_table(table1_, table2_)
# check_table(table2_, table3_)
#
#
#
# np.testing.assert_almost_equal(table2_, table2)
# np.testing.assert_almost_equal(table3_, table3)
#
# # and now verify that table 2 and table 3 are coherent
# # for an entry in the table, we can compute its position in the next grid
#
# shift = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
# for i in range(len(table2)):
#     for j in range(8):
#         if table2[i, j] != -1:
#             # table2[i, j]
#             xyz = table2[i, -3:] * 2
#             idx_next = int(table2[i, j])
#             xyz_next = table3[idx_next, -3:]
#             xyz_next_from_prev = xyz + shift[j]
#             np.testing.assert_almost_equal(xyz_next_from_prev, xyz_next)
#
#
#
#
#
# #check that xyz corresponds to the same order as indexing
# grid = g3
# grid_000 = grid[0::2, 0::2, 0::2]
# grid_001 = grid[0::2, 0::2, 1::2]
# grid_010 = grid[0::2, 1::2, 0::2]
# grid_100 = grid[1::2, 0::2, 0::2]
# grid_011 = grid[0::2, 1::2, 1::2]
# grid_110 = grid[1::2, 1::2, 0::2]
# grid_101 = grid[1::2, 0::2, 1::2]
# grid_111 = grid[1::2, 1::2, 1::2]
#
# idx = jnp.arange(0, grid.sum())
# tmp = np.ones_like(grid) * -1
# tmp[grid] = idx
# mask_tmp = tmp != -1
# mask_tmp = mask_tmp == grid
# mask_tmp.all() # true
# grid = tmp
# grid_000 = grid[0::2, 0::2, 0::2]
# grid_001 = grid[0::2, 0::2, 1::2]
# grid_010 = grid[0::2, 1::2, 0::2]
# grid_100 = grid[1::2, 0::2, 0::2]
# grid_011 = grid[0::2, 1::2, 1::2]
# grid_110 = grid[1::2, 1::2, 0::2]
# grid_101 = grid[1::2, 0::2, 1::2]
# grid_111 = grid[1::2, 1::2, 1::2]
#
#
# table2[:, 0] = f_000
# table2[:, 1] = f_001
# table2[:, 2] = f_010
# table2[:, 3] = f_100
# table2[:, 4] = f_011
# table2[:, 5] = f_110
# table2[:, 6] = f_101
# table2[:, 7] = f_111
#
#
# # are the x, y, z in the same order as in g2.flatten()
# # easy test
# tmp1 = np.random.random(g2.shape)
# tmp2 = np.zeros_like(tmp1)
# tmp2[x, y, z] = tmp1[x, y, z]
# tmp3 = tmp2 > 0
# tmp4 = tmp3.flatten() == flat
#
#
#
# mask1 = table[:,:8] != -1
# mask2 = table2[:,:8] != -1
# bool = mask1 == mask2
# bool.all()
#
# (f_000 != -1).sum() + (f_001 != -1).sum() + (f_010 != -1).sum() + (f_100 != -1).sum() + (f_110 != -1).sum() + (f_101 != -1).sum() + (f_111 != -1).sum() + (f_011 != -1).sum()
#
# table = np.ones([flat.sum(), 8+3]) * -1
# idx = jnp.arange(0, g3.sum())
#
# x, y, z = np.where(g2) # * side of cube
# # are the x, y, z in the same order as in g2.flatten()
# # easy test
# tmp1 = np.random.random(g2.shape)
# tmp2 = np.zeros_like(tmp1)
# tmp2[x, y, z] = tmp1[x, y, z]
# tmp3 = tmp2 > 0
# tmp4 = tmp3.flatten() == flat
#
# table[:, 8] = x
# table[:, 9] = y
# table[:, 10] = z
#
# prev = 0
# next = f_000.sum()
# table[f_000, 0] = idx[prev: next]
# prev = next
# next += f_001.sum()
# table[f_001, 1] = idx[prev: next]
# prev = next
# next += f_010.sum()
# table[f_010, 2] = idx[prev: next]
# prev = next
# next += f_100.sum()
# table[f_100, 3] = idx[prev: next]
# prev = next
# next += f_011.sum()
# table[f_011, 4] = idx[prev: next]
# prev = next
# next += f_110.sum()
# table[f_110, 5] = idx[prev: next]
# prev = next
# next += f_101.sum()
# table[f_101, 6] = idx[prev: next]
# prev = next
# next += f_111.sum()
# table[f_111, 7] = idx[prev: next]
#
#
#
# grid = g4
# grid_000 = grid[0::2, 0::2, 0::2]
# grid_001 = grid[0::2, 0::2, 1::2]
# grid_010 = grid[0::2, 1::2, 0::2]
# grid_100 = grid[1::2, 0::2, 0::2]
# grid_011 = grid[0::2, 1::2, 1::2]
# grid_110 = grid[1::2, 1::2, 0::2]
# grid_101 = grid[1::2, 0::2, 1::2]
# grid_111 = grid[1::2, 1::2, 1::2]
#
# flat = g3.flatten()
# f_000 = grid_000.flatten()
# f_001 = grid_001.flatten()
# f_010 = grid_010.flatten()
# f_100 = grid_100.flatten()
# f_011 = grid_011.flatten()
# f_110 = grid_110.flatten()
# f_101 = grid_101.flatten()
# f_111 = grid_111.flatten()
#
# f_000 = f_000[flat]
# f_001 = f_001[flat]
# f_010 = f_010[flat]
# f_100 = f_100[flat]
# f_011 = f_011[flat]
# f_110 = f_110[flat]
# f_101 = f_101[flat]
# f_111 = f_111[flat]
#
# (f_000).sum() + (f_001).sum() + (f_010).sum() + (f_100).sum() + (f_110).sum() + (f_101).sum() + (f_111).sum() + (f_011).sum()
#
#
# # we want a function that takes table, the valuesin the columns, and verifies if
# # computes the permutation of indices in the next order.
# # what if we build another table, with the new xyz values, then compute the "correspondence"
# # in table 1 the xyz are the coarse voxel indices
# # in table 2 the xyz are the fine voxel indices of the prev table
# # if we just build the new table 2, which we can do, then we will have to permute it
#
#
# prev_i = 0
# next_i = f_000.sum()
# idx_000 = jnp.arange(prev_i, next_i)
#
# prev_i = next_i
# next_i += f_001.sum()
# idx_001 = jnp.arange(prev_i, next_i)
#
# prev_i = next_i
# next_i += f_010.sum()
# idx_010 = jnp.arange(prev_i, next_i)
#
# prev_i = next_i
# next_i += f_100.sum()
# idx_100 = jnp.arange(prev_i, next_i)
#
# prev_i = next_i
# next_i += f_011.sum()
# idx_011 = jnp.arange(prev_i, next_i)
#
# prev_i = next_i
# next_i += f_110.sum()
# idx_110 = jnp.arange(prev_i, next_i)
#
# prev_i = next_i
# next_i += f_101.sum()
# idx_101 = jnp.arange(prev_i, next_i)
#
# prev_i = next_i
# next_i += f_111.sum()
# idx_111 = jnp.arange(prev_i, next_i)
#
# table2[f_000, 0] = idx_000
# table2[f_001, 1] = idx_001
# table2[f_010, 2] = idx_010
# table2[f_100, 3] = idx_100
# table2[f_011, 4] = idx_011
# table2[f_110, 5] = idx_110
# table2[f_101, 6] = idx_101
# table2[f_111, 7] = idx_111
#
# # remove the dead voxels and the coarser level
# tmp = table2[f2]
#
# tmp = tmp > -1
# tmp.sum()
#
#
#
# tmp = f_101 * f2
# tmp.sum()
# f_101.sum()
#
# (f_000).sum() + (f_001).sum() + (f_010).sum() + (f_100).sum() + (f_110).sum() + (f_101).sum() + (f_111).sum() + (f_011).sum()
# len(idx_000) + len(idx_001) + len(idx_010) + len(idx_100) + len(idx_011) + len(idx_110) + len(idx_101) + len(idx_111)
# (f_000* f2).sum() + (f_001* f2).sum() + (f_010* f2).sum() + (f_100* f2).sum() + (f_110* f2).sum() + (f_101* f2).sum() + (f_111* f2).sum() + (f_011* f2).sum()
#
#
# def quotient_mat(grid):
#     grid_000 = grid[0::2, 0::2, 0::2]
#     grid_001 = grid[0::2, 0::2, 1::2]
#     grid_010 = grid[0::2, 1::2, 0::2]
#     grid_100 = grid[1::2, 0::2, 0::2]
#     grid_011 = grid[0::2, 1::2, 1::2]
#     grid_110 = grid[1::2, 1::2, 0::2]
#     grid_101 = grid[1::2, 0::2, 1::2]
#     grid_111 = grid[1::2, 1::2, 1::2]
#
#
#
# # what is a good test?
#
#
#
# grid3 = mipmap[3]
# grid4 = mipmap[4]
# grid5 = mipmap[5]
# grid3 = grid3 > 0
# grid4 = grid4 > 0
# grid5 = grid5 > 0
# tmp = np.zeros_like(grid4)
# for i in range(32):
#     for j in range(32):
#         for k in range(32):
#             tmp[i,j,k]= grid5[2*i:2*i+2, 2*j:2*j+2, 2*k:2*k+2].any()
#
# tmp2 = tmp == grid4
# tmp2.all()
#
# grid4_000 = grid4[0::2, 0::2, 0::2].flatten()
# grid4_001 = grid4[0::2, 0::2, 1::2].flatten()
# grid4_010 = grid4[0::2, 1::2, 0::2].flatten()
# grid4_100 = grid4[1::2, 0::2, 0::2].flatten()
# grid4_011 = grid4[0::2, 1::2, 1::2].flatten()
# grid4_110 = grid4[1::2, 1::2, 0::2].flatten()
# grid4_101 = grid4[1::2, 0::2, 1::2].flatten()
# grid4_111 = grid4[1::2, 1::2, 1::2].flatten()
#
# grid4_list = []
# grid4_list.append(grid4_000)
# grid4_list.append(grid4_001)
# grid4_list.append(grid4_010)
# grid4_list.append(grid4_100)
# grid4_list.append(grid4_011)
# grid4_list.append(grid4_110)
# grid4_list.append(grid4_101)
# grid4_list.append(grid4_111)
#
# sum = 0
# for i in range(8):
#     sum += grid4_list[i].sum()
#
# tmp = grid4_000 | grid4_001 | grid4_010 | grid4_100 | grid4_011 | grid4_110 | grid4_101 | grid4_111
# tmp = np.zeros([16*16*16, 2,2,2])
# tmp[:, 0, 0, 0] = grid4_000
# tmp[:, 0, 0, 1] = grid4_001
# tmp[:, 0, 1, 0] = grid4_010
# tmp[:, 1, 0, 0] = grid4_100
# tmp[:, 0, 1, 1] = grid4_011
# tmp[:, 1, 0, 1] = grid4_101
# tmp[:, 1, 1, 1] = grid4_111
#
# grid_finer = tmp > 0
# grid_coarse = grid3.flatten()
#
# tmp_finer = jnp.where(grid_finer)
#
#
#
#
#
# idx3 = jnp.where(grid3_)[0]
# idx4 = jnp.where(grid4.flatten())[0]
# table3 = np.zeros([len(idx3), 8])
# table4 = np.zeros([len(idx4), 8])
#
# idx = np.arange(len(idx3), len(idx3)+grid4_000.sum())
# table3[grid4_000, 0] = idx
#
#
#
#
# tmp2 = tmp == grid3
#
#
#
# # question: how do you reshape
#
#
#
# tmp = grid5.reshape([32*32*32, 2, 2, 2])
# grid4 = grid4.flatten()
# tmp4 = grid4 > 0
# tmp = tmp > 0
# tmp = tmp.any(axis=[1,2,3])
#
#
#
# fifo = []
# grid = mipmap[0]
# grid_next = mipmap[1]
# level = 0
# counter = 0
# for i in range(2**(level+1)):
#     for j in range(2**(level+1)):
#         for k in range(2**(level+1)):
#             if grid[i, j, k] > 0:
#                 xyz =
#
#                 node = ONode(level, )
#
#
# idx_list = []
# for i in range(8):
#     tmp = mipmap[i].flatten()
#     idx = jnp.where(tmp)[0]
#     idx_list.append(idx)
#
# node_nb = 0
# for i in range(8):
#     tmp = mipmap[i] > 0
#     node_nb += tmp.sum()
#
# mipmap.reverse()
# grid = mipmap[0]
# x, y, z = jnp.where(grid)
#
#
# tmp = mipmap[0].flatten()
# idx = jnp.where(tmp)[0]
#
# idx = jnp.indices((2, 2, 2))
#
#
# graph_table = jnp.ones([node_nb+1, 8+1+3])*-1
# current_node = 0
# current_xyz = np.zeros(3)
# level = 0
#
# root = 0
# current_idx = 0
# grid = mipmap[0].flatten()
# tmp = jnp.where(grid)[0]
#
# grid = mipmap[1].flatten()
# tmp = jnp.where(grid)[0]
#
# graph_table[root, 1] = mipmap
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
# ori = np.zeros(3)
# dir = np.ones(3) / np.sqrt(3)
# inv_dir = 1/dir
#
# # take the ray,
# intersection_levels = []
# # create a list of a
#
# total_width = 256
# bbox = create_bbox(total_width)
# level = 4
# nb_bins = 256 / 2**level
# assert nb_bins == len(mipmap[level])
# box_width = 2**level
#
# # class OctreeNode:
# #     def __init__(self):
# #         self.origin = np.zeros(3)
# #         self.level = 0
# #         self.children = []
# #         self.is_leaf = False
#
# stack = [mipmap[7]] # the coarsest grid
# stack_level = [7]
# stack_idx = [np.zeros(3)]
#
# grid = mipmap[0] # the initial grid (finest)
#
# # we actually want the indices in the final grid
# intersected_nodes = []
# while stack:
#     node = stack.pop()
#     node_level = stack_level.pop()
#     node_idx = stack_idx.pop()
#
#     if node_level == 0 & grid[node_idx] > 0:
#         intersected_nodes.append(node_idx)
#     else:
#         for i in range(2):
#             for j in range(2):
#                 for k in range(2):
#                     if node[i, j, k] != 0:
#                         bottom = np.array([i * box_width, j * box_width, k * box_width])
#                         top = bottom + box_width
#                         t1, t2 = intersect_ray_aabb(ori, inv_dir, bottom, top)
#                         if t1 < t2: # then we want to explore this node
#                             stack.append()
#                             stack_level.append()
#                             stack_idx.append()
#
#
# import jax.numpy as jnp
# from jax import grad, jit, vmap
# from jax import random
# import numpy as np
# import jax.numpy as jnp
# from jax import lax
# arr = np.array([3, 2, 0, 0, 1])
# print(np.sort(arr))
#
#
# import numpy as np
# import jax.numpy as jnp
# from jax import lax
# arr = jnp.array([3, 2, 0, 0, 1])
#
# print(lax.sort(arr))
# print(jnp.sort(arr))
#
# key = random.PRNGKey(0)
# size = 256*256*256
# x = random.randint(key, (size, 1), minval=0, maxval=1000000)
# import time
# t0 = time.time()
# y = jnp.sort(x)
# print(time.time() - t0)
#
#
# @jit
# def fct(input, x):
#     input = input + x
#     return jnp.sort(input)
#
# x = jnp.arange(100)
# key = random.PRNGKey(0)
# size = 256*256*256
# input = random.randint(key, (size, 1), minval=0, maxval=1000000)
#
# t0 = time.time()
# res_vmap = vmap(fct, in_axes=(None, 0))(input, x)
# print(time.time() - t0)