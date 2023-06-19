import numpy as np
import unittest
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import device_put
import jax
import time
import contextlib
import timeit
import copy
import time

from jax_mipmap import *
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

class TestJax(unittest.TestCase):
    def setUp(self):
        self.size = 256
        self.tmin = random.normal(random.PRNGKey(0), (100,))
        self.diff = jnp.abs(random.normal(random.PRNGKey(1), (100,)))
        self.tmax = self.tmin + self.diff
        self.spacing = 0.01
        self.ar = []
        for i in range(100):
            self.ar.append(jnp.arange(self.tmin[i], self.tmax[i], self.spacing))

    def test_create_occupancy_voxels(self):
        mat = np.zeros([4, 4, 4])
        cube = np.ones([2, 2, 2])
        mat[:2, :2, :2] = cube
        mat[2:, 2:, 2:] = cube

        occupancy_mat = np.zeros(mat.shape)
        occupancy_mat[mat > 0] = 1
        data = jnp.array(occupancy_mat.astype(jnp.float32))
        data = data[None, :, :, :, None]

        kernel = jnp.ones([2, 2, 2])[:, :, :, jnp.newaxis, jnp.newaxis]
        dn = lax.conv_dimension_numbers(data.shape, kernel.shape, ('NHWDC', 'HWDIO', 'NHWDC'))

        out = lax.conv_general_dilated(data,  # lhs = image tensor
                                       kernel,  # rhs = conv kernel tensor
                                       (1, 1, 1),  # window strides
                                       'SAME',  # padding mode
                                       (1, 1, 1),  # lhs/image dilation
                                       (1, 1, 1),  # rhs/kernel dilation
                                       dn)  # dimension_numbers
        out = out[0, :, :, :, 0]
        mask = out == 8
        mask = np.array(mask)
        res = np.zeros_like(mask)
        res[mask] = 1
        assert res[0,0,0] == 1
        assert res[2,2,2] == 1

    def test_check_table(self):
        # What this does is that the first 8 entries of a row denote the index of the row in the next table.
        # If a column is -1 for that row, it means that there is not leaf/intersection.
        # For a given row, the last 3 indices denote the position of the upper corner of the current voxel.
        pos1 = np.array([0,0,0])
        offset = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
        next_idx = np.array([0,1,2,3,4,5,6,7])
        table1 = np.zeros([1,11])
        table1[0, 8:] = pos1
        table1[0, :8] = next_idx
        table2 = np.ones([8, 11]) * -1
        table2[:, 8:] = pos1 + offset
        assert check_table(table1, table2)

    def test_table_last(self):
        grid = np.zeros([2,2,2], dtype=int)
        grid[0,0,0] = 1
        grid[0, 1, 0] = 1
        grid[1,0,1] = 1
        grid = grid.astype(bool)
        x, y, z = np.where(grid)

        # This is to understand the ordering from np.where
        idx = jnp.arange(0, grid.sum())
        tmp = np.ones_like(grid) * -1
        tmp[grid] = idx
        grid = tmp
        for i in range(len(x)):
            assert grid[x[i], y[i], z[i]] == i

        table = np.ones([len(x), 3]) * -1
        table[0, -3:] = [0,0,0]
        table[1, -3:] = [0,1,0]
        table[2, -3:] = [1,0,1]

        grid = np.zeros([2,2,2])
        grid[0,0,0] = 1
        grid[0, 1, 0] = 1
        grid[1,0,1] = 1

        table2 = make_table_last(grid)
        np.testing.assert_almost_equal(table, table2)

    def test_mipmap_compute(self):
        grid_fine = np.zeros([4,4,4], dtype=bool)
        grid_fine[0, 0, 0] = True
        grid_fine[0, 1, 1] = True
        grid_fine[2, 2, 1] = True
        grid_fine[3, 3, 1] = True
        grid_fine[3, 0, 3] = True
        grid_coarse = np.zeros([2,2,2], dtype=bool)
        grid_coarse[0, 0, 0] = True
        grid_coarse[1, 1, 0] = True
        grid_coarse[1, 0, 1] = True
        mipmap = mipmap_compute(grid_fine)
        res = np.array(mipmap[1])
        res = res > 0
        np.testing.assert_almost_equal(res, grid_coarse)

    def test_make_table(self):
        grid = np.random.binomial(n=1, p=0.1, size=[16**3])
        grid = grid.reshape([16, 16, 16])
        mipmap = mipmap_compute(grid)
        mipmap.reverse()

        g0 = mipmap[0] > 0
        g1 = mipmap[1] > 0
        g2 = mipmap[2] > 0
        g3 = mipmap[3] > 0
        g4 = mipmap[4] > 0

        t01 = make_table(g0, g1)
        t12 = make_table(g1, g2)
        t23 = make_table(g2, g3)
        t34 = make_table(g3, g4)

        check_table(t01, t12)
        check_table(t12, t23)
        check_table(t23, t34)

    def test_mipmap_viz(self):
        path = '/home/adrian/Code/svox2/opt/ckpt/exp2/ckpt.npz'
        data = np.load(path, allow_pickle=True)
        npy_links = data['links']
        npy_density_data = data['density_data']
        npy_sh_data = data['sh_data']

        # hack
        npy_density_data[0] = 0
        npy_sh_data[0] = 0
        npy_links[npy_links < 0] = 0

        density_matrix = np.squeeze(npy_density_data[npy_links.clip(min=0)])
        density_matrix[density_matrix < 0.2] = 0
        mask_sphere = filter_over_sphere(density_matrix, np.ones(3) * 128, 100)
        density_matrix[~mask_sphere] = 0

        occupancy_voxels = create_occupancy_voxels(density_matrix)

        # Mipmap
        mipmap = mipmap_compute(occupancy_voxels)
        mipmap.reverse()

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

        ray_dir = np.array([1, 0.8, 0.8])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        ray_inv_dir = 1 / ray_dir
        ray_ori = np.ones(3) * 0

        sort_pt = np.array([0, 3, -10])
        max_level = 3

        out1 = ray_mipmap_intersect_forloop(ray_ori, ray_inv_dir, tables_list, max_level)
        out1 = np.array(out1)
        dist = np.linalg.norm(out1 - sort_pt, axis=1)
        out1 = out1[dist.argsort()]

        geoms = viz_mipmap(mipmap, max_level, 256, np.zeros(3), ray_dir * 400, out1)
        import open3d.visualization as vis
        vis.draw(geoms)


    def test_mipmap_intersect(self):
        path = '/home/adrian/Code/svox2/opt/ckpt/exp2/ckpt.npz'
        data = np.load(path, allow_pickle=True)
        npy_links = data['links']
        npy_density_data = data['density_data']
        npy_sh_data = data['sh_data']

        # hack
        npy_density_data[0] = 0
        npy_sh_data[0] = 0
        npy_links[npy_links < 0] = 0

        density_matrix = np.squeeze(npy_density_data[npy_links.clip(min=0)])
        density_matrix[density_matrix < 0.2] = 0
        mask_sphere = filter_over_sphere(density_matrix, np.ones(3) * 128, 100)
        density_matrix[~mask_sphere] = 0

        occupancy_voxels = create_occupancy_voxels(density_matrix)

        # Mipmap
        mipmap = mipmap_compute(occupancy_voxels)
        mipmap.reverse()

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

        ray_dir = np.array([1, 0.8, 0.8])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        ray_inv_dir = 1 / ray_dir
        ray_ori = np.ones(3) * 0

        sort_pt = np.array([0, 3, -10])
        max_level = 5
        table = make_table_last(g5)
        out_a = ray_grid_intersect(ray_ori, ray_inv_dir, table, max_level)
        out_a = np.array(out_a)
        dist = np.linalg.norm(out_a - sort_pt, axis=1)
        out_a = out_a[dist.argsort()]

        check_in_grid(out_a, g5)

        out1 = ray_mipmap_intersect_forloop(ray_ori, ray_inv_dir, tables_list, max_level)
        out1 = np.array(out1)
        dist = np.linalg.norm(out1 - sort_pt, axis=1)
        out1 = out1[dist.argsort()]

        check_in_grid(out1, g5)

        out3 = ray_mipmap_intersect_vectorized(ray_ori, ray_inv_dir, tables_list, max_level)  # incorrect
        out3 = np.array(out3)
        dist = np.linalg.norm(out3 - sort_pt, axis=1)
        out3 = out3[dist.argsort()]

        check_in_grid(out3, g5)

        np.testing.assert_almost_equal(np.array(out_a), np.array(out1))
        np.testing.assert_almost_equal(np.array(out3), np.array(out1))


if __name__ == '__main__':
    unittest.main()