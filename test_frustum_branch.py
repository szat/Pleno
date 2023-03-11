import numpy as np
import unittest
from scipy.special import *

from frustum_branch import *
from spherical_harmonics import sh_cartesian


class TestArchitecture(unittest.TestCase):
    def setUp(self):
        self.idim = 100
        # So the grid looks like [0.0, 1.0, ... , 100.0]^3, functions take (dx,dy,dz)=(1,1,1) by default
        self.nb_rays = 7
        self.nb_samples = 11
        # 9 harmonic coeficients per voxel
        self.grid = np.random.rand(self.idim, self.idim, self.idim, 9)
        self.opacity = np.random.rand(self.idim, self.idim, self.idim)

    def test_rays_to_frustrum_shape(self):
        # shifts/scalings are just for testing and making sure that random tesing samples does not overflow grid:
        ray_origins = np.random.rand(self.nb_rays, 3) * 20 + self.idim/2
        ray_dir_vecs = np.random.rand(self.nb_rays, 3) + 0.5
        ray_samples = np.random.rand(self.nb_rays, self.nb_samples) * 30
        frustum, dir_vec_neighs = rays_to_frustrum(ray_origins, ray_dir_vecs, ray_samples)
        np.testing.assert_array_almost_equal(frustum.shape,
                                             np.array([self.nb_rays, self.nb_samples, 8, 3]))
        np.testing.assert_array_almost_equal(dir_vec_neighs.shape,
                                             np.array([self.nb_rays, self.nb_samples, 8, 3]))

    def test_frustrum_to_harmonics_shape(self):
        # shifts/scalings are just for testing and making sure that random tesing samples does not overflow grid:
        ray_origins = np.random.rand(self.nb_rays, 3) * 20 + self.idim/2
        ray_dir_vecs = np.random.rand(self.nb_rays, 3) + 0.5
        ray_samples = np.random.rand(self.nb_rays, self.nb_samples) * 30
        frustum, dir_vec_neighs = rays_to_frustrum(ray_origins, ray_dir_vecs, ray_samples)
        neighs_harmonics, neighs_opacities = frustrum_to_harmonics(frustum, dir_vec_neighs, self.grid, self.opacity)
        np.testing.assert_array_almost_equal(neighs_harmonics.shape,
                                             np.array([self.nb_rays, self.nb_samples, 8, 9]))
        np.testing.assert_array_almost_equal(neighs_opacities.shape,
                                             np.array([self.nb_rays, self.nb_samples, 8]))

    def test_rays_to_frustum_easy_values(self):
        nr_rays = 4
        nr_samples = 3
        root3 = np.sqrt(3)
        ray_origins = np.ones((nr_rays, 3)) * 50
        ray_dir_vecs = np.array([[1, 0, 0], # pointing to +x
                                 [0, 1, 0], # pointing to +y
                                 [0, 0, 1], # pointing to +z
                                 [1, 1, 1]] # pointing to (1,1,1)
                               ).reshape((nr_rays, 3))
        # samples defined by following distances from ray origin along ray dir:
        ray_samples = np.stack([np.array([2.5, 5.7, 8.5]),
                                np.array([3.3, 4.9, 9.2]),
                                np.array([3.5, 5.7, 7.5]),
                                np.array([1.5*root3, 2.2*root3, 5.8*root3])]
                              ).reshape((nr_rays, nr_samples))

        # test only first neighbour (relative (0, 0, 0)-neighbour)
        voxels_ijk = np.array([
                                [[52, 50, 50], [55, 50, 50], [58, 50, 50]],
                                [[50, 53, 50], [50, 54, 50], [50, 59, 50]],
                                [[50, 50, 53], [50, 50, 55], [50, 50, 57]],
                                [[51, 51, 51], [52, 52, 52], [55, 55, 55]]
                              ])
        dir_vecs = np.array([
                              [[2, 0, 0], [5, 0, 0], [8, 0, 0]],
                              [[0, 3, 0], [0, 4, 0], [0, 9, 0]],
                              [[0, 0, 3], [0, 0, 5], [0, 0, 7]],
                              [[1, 1, 1], [2, 2, 2], [5, 5, 5]]
                            ])

        assert voxels_ijk.shape == (nr_rays, nr_samples, 3)
        frustum, dir_vec_neighs = rays_to_frustrum(ray_origins, ray_dir_vecs, ray_samples)
        np.testing.assert_array_almost_equal(frustum[:, :, 0, :],
                                             voxels_ijk)
        np.testing.assert_array_almost_equal(dir_vec_neighs[:, :, 0, :],
                                             dir_vecs)

    def test_frustum_to_harmonics_easy_values(self):
        nr_rays = 1
        nr_samples = 1
        ray_origins = np.array([50.5, 50.5, 50.5]).reshape((nr_rays, 3))
        ray_dir_vecs = np.array([1, 0, 0]).reshape((nr_rays, 3)) # pointing to +x
        ray_samples = np.array([0]).reshape((nr_rays, nr_samples)) # sample at ray origin
        frustum, dir_vec_neighs = rays_to_frustrum(ray_origins, ray_dir_vecs, ray_samples)
        ret_harmonics, ret_opacities = frustrum_to_harmonics(frustum, dir_vec_neighs, self.grid, self.opacity)

        true_opacities = np.array([self.opacity[50, 50, 50], self.opacity[51, 50, 50],
                                   self.opacity[50, 51, 50], self.opacity[51, 51, 50],
                                   self.opacity[50, 50, 51], self.opacity[51, 50, 51],
                                   self.opacity[50, 51, 51], self.opacity[51, 51, 51]])
        true_opacities = true_opacities.reshape((nr_rays, nr_samples, 8))
        assert ret_opacities.shape == true_opacities.shape
        np.testing.assert_array_almost_equal(ret_opacities, true_opacities)

        coeff_harmonics = np.stack([self.grid[50, 50, 50], self.grid[51, 50, 50],
                                    self.grid[50, 51, 50], self.grid[51, 51, 50],
                                    self.grid[50, 50, 51], self.grid[51, 50, 51],
                                    self.grid[50, 51, 51], self.grid[51, 51, 51]])
        coeff_harmonics = np.expand_dims(coeff_harmonics, axis=(0, 1)) # 1 x 1 x 8 x 9
        true_harmonics = sh_cartesian(np.array([[-1, -1, -1], [1, -1, -1],
                                                [-1, 1, -1], [1, 1, -1],
                                                [-1, -1, 1], [1, -1, 1],
                                                [-1, 1, 1], [1, 1, 1]]
                                              ).reshape((8, 3)))
        true_weighted_harmonics = coeff_harmonics * true_harmonics.reshape((1, 1, 8, 9))
        assert ret_harmonics.shape == true_weighted_harmonics.shape
        np.testing.assert_array_almost_equal(ret_harmonics, true_weighted_harmonics)


if __name__=='__main__':
    unittest.main()
